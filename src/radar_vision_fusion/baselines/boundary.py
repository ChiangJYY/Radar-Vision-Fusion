"""Linear, MLP and vanilla GAT depth-residual architectures, without training orchestration."""
import torch
from torch import nn
from torch.nn import functional as F

class LinearResidualNet(nn.Module):
    """
    Δz = w_u · u_n + w_v · v_n + w_z · z_n + b
    4 个参数。后续实验会升级到 MLP / GNN。
    """
    def __init__(self, input_dim=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

        # 初始化: Δz 接近 0，让训练从一个稳定的 baseline 起步
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

class MLPResidualNet(nn.Module):
    """
    Δz = MLP([u_n, v_n, z_n])
    
    与 Linear (B0) 的区别:
    - 两层隐藏层 + ReLU + LayerNorm（稳定训练）
    - Xavier 初始化
    - 可选 dropout
    """
    def __init__(self, input_dim=3, hidden_dims=[64, 32], dropout=0.0):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)

class VanillaGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=4, heads=4, dropout=0.0, concat=True):
        super().__init__()
        assert out_dim % heads == 0
        self.heads   = heads
        self.head_dim = out_dim // heads
        self.concat  = concat
        self.dropout = dropout

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, self.head_dim))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr=None):
        """
        x:           (N, in_dim)
        edge_index:  (2, E)
        edge_attr:   (E, edge_dim) — 生成但不使用（与 SimpleGNN 接口兼容）
        """
        N = x.size(0)
        H, D = self.heads, self.head_dim

        Wh = self.W(x).view(N, H, D)
        src, dst = edge_index[0], edge_index[1]
        Wh_src = Wh[src]
        Wh_dst = Wh[dst]

        # 注意力打分：无边特征注入
        e = (Wh_src * self.a_src).sum(-1) + (Wh_dst * self.a_dst).sum(-1)
        e = self.lrelu(e)

        # softmax over src
        e_max = torch.full((N, H), -1e9, dtype=e.dtype, device=x.device)
        e_max = e_max.scatter_reduce(0, src.unsqueeze(-1).expand(-1, H), e,
                                      reduce="amax", include_self=False)
        e_max_per_edge = e_max[src]
        e_shift = e - e_max_per_edge
        exp_e = torch.exp(e_shift)

        denom = torch.zeros((N, H), dtype=e.dtype, device=x.device)
        denom.scatter_add_(0, src.unsqueeze(-1).expand(-1, H), exp_e)
        denom = denom.clamp(min=1e-12)
        alpha = exp_e / denom[src]

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        msg = Wh_dst * alpha.unsqueeze(-1)
        out = torch.zeros((N, self.heads * D), dtype=msg.dtype, device=x.device)
        out_v = out.view(N, H, D)
        out_v.scatter_add_(0, src.view(-1, 1, 1).expand(-1, H, D), msg)
        return out_v.view(N, H * D) if self.concat else out_v.mean(dim=1)


class VanillaGATResidualNet(nn.Module):
    def __init__(self, in_dim=3, edge_dim=4, hidden=32, heads=4, layers=2, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.norms_in = nn.LayerNorm(hidden)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(VanillaGATLayer(
                hidden, hidden, edge_dim=edge_dim,
                heads=heads, dropout=dropout, concat=True))
            self.norms.append(nn.LayerNorm(hidden))

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, edge_index, edge_attr=None):
        """
        x:           (N, 3)   节点特征
        edge_index:  (2, E)
        edge_attr:   (E, 4)   — 生成但不使用
        返回: Δz     (N,)
        """
        h = self.input_proj(x)
        h = self.norms_in(h)
        h = F.relu(h)
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, edge_index, edge_attr)
            h = norm(h + h_new)
            h = F.relu(h)
        return self.head(h).squeeze(-1)
