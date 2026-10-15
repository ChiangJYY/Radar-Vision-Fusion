# From Image-Defined Regions to Metric Boundaries: Radar--Vision Fusion for Railway Pedestrian Intrusion Monitoring Under Degraded Visibility

面向低能见度条件下铁路行人入侵监测的雷达与视觉融合研究。本页分别介绍配套方法代码与原始数据集。

## 目录

- [公开代码](#公开代码)：[结构](#项目结构) · [安装](#安装) · [运行](#运行方法) · [采集工具](#数据采集工具)
- [公开数据集](#公开数据集)：[采集平台](#采集平台) · [下载](#下载与存储) · [数据格式](#数据格式) · [同步与标定](#时间同步与坐标标定)
- [第三方与许可](#第三方与许可)

## 公开代码

提供 **ECDR-GAT** 区域边界重建、**Trans-DKF** 自适应融合，以及视觉观测、数据处理、标定、多目标跟踪、越界判断、基线和评测工具。学习模型推理需自行准备兼容权重；本仓库不提供完整论文复现工程。

### 项目结构

```text
├── src/
│   ├── radar_vision_fusion/   # 核心方法、感知、标定、跟踪和评测
│   └── third_party/          # RT-DETR、Depth Anything V2 源码子集
├── scripts/                  # 数据转换、标定、推理、评测及绘图
├── tools/                    # camera、radar、uwb 采集工具
├── configs/                  # 推理配置与空白标定模板
├── dataset/                  # 数据存放位置，仓库不附带原始数据
├── figures/                  # README 图片
├── docs/                     # 附加文档预留
├── requirements.txt
└── pyproject.toml
```

### 安装

建议 Python 3.10+，使用独立环境，先安装与本机 CPU/CUDA 匹配的 PyTorch 和 torchvision，再在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` 给出版本下限，并非锁定环境；工业相机还需单独安装 MVS SDK。

### 运行方法

在仓库根目录执行，尖括号内容替换为实际路径。输入数据、区域和标定需由使用者提供，输出位置通过参数指定。

```bash
# 基础跟踪：不需要学习模型权重
python scripts/run_inference.py --input <detections.json> --region-file <region.json> --mode fixed --output <tracks.json>

# Trans-DKF 融合跟踪与越界监测
python scripts/run_inference.py --input <detections.json> --region-file <region.json> --mode learned --checkpoint <transdkf.pt> --output <tracks.json>

# ECDR-GAT 区域重建
python scripts/run_boundary.py --input <boundary.csv> --checkpoint <ecdr.pt> --calibration <calibration.json> --output <boundary.json>
```

| 脚本 | 用途 |
| --- | --- |
| `run_visual.py` | RT-DETRv2 与度量 Depth Anything V2 生成视觉观测 |
| `prepare_sequence.py` | 相机与雷达数据转换、时间对齐 |
| `calibrate_extrinsics.py` | 雷达与相机外参标定 |
| `evaluate_results.py` | 区域与跟踪评测，需提供参考真值 |
| `plot_results.py` | 结果绘图 |

各脚本可通过 `python scripts/<脚本名> --help` 查看参数。`fixed` 为基础跟踪模式，不代表 Trans-DKF 性能。

### 数据采集工具

以下工具用于自行采集数据，输出路径按需替换。

#### 摄像头

使用 `tools/camera/capture_camera.py`，需安装相机驱动与 MVS SDK，并指定 SDK 的 `MvImport` 目录：

```bash
python tools/camera/capture_camera.py --sdk-path "<MVS的MvImport目录>" --output-dir "<相机输出目录>" --fps 10 --width 2448 --height 2048 --exposure-ms 50
```

界面中点击“开始”保存、“停止”暂停保存；曝光可在界面调整。程序按采集会话建立子目录，以 `YYYYMMDD_HHMMSS_mmm.bmp` 保存图像，时间戳来自采集电脑本地时钟。上述参数可按设备与场景调整。

#### 雷达

使用 `tools/radar/Visualizer/gui_main.py`，依赖 PyQt5、pyserial、pyqtgraph 和 PyOpenGL，已列入 `requirements.txt`：

```bash
python tools/radar/Visualizer/gui_main.py --output-dir "<雷达输出目录>"
```

界面中选择 CLI/Data 串口与 **Long Range People Detection** 模式，点击 **Connect**，通过 **Select Configuration** 选择与设备固件匹配的 `.cfg`（位于 `tools/radar/Visualizer/chirp_configs/`），再点击 **Start and Send Configuration**。使用 **Start Save / Stop Save** 控制文本记录，输出 `LRPD_data_YYYYMMDD_HHMMSS.txt`，包含时间戳、点云和目标轨迹。

UWB 采集使用 `tools/uwb/` 中的 Windows 厂商上位机，日志保存位置在其界面设置。各设备记录需按时间戳对齐，不能按文件序号直接配对。

## 公开数据集

### 数据概况

配套原始数据包含摄像头图像、毫米波雷达点云及设备跟踪输出、UWB 参考定位日志，覆盖操场、草地和铁路场景中的单人、多人、雾天、夜间及边界行走工况。

当前数据包包含 **13 个工况目录、67,864 张 BMP 图像**。这是原始文件统计，不等同于论文处理后的同步帧数或实验划分；数据包不附带训练数据划分、模型权重、缓存及逐帧预测。

### 采集平台

按论文 **Experimental Setup** 部分，平台配置如下：

| 设备 | 配置与作用 |
| --- | --- |
| 毫米波雷达 | Texas Instruments IWR6843，提供目标位置、速度等观测 |
| 工业相机 | Hikvision MV-CS050-10UC |
| 相机镜头 | 8 mm，MVL-MF0824M-5MPE |
| 图像分辨率 | 2448 × 2048 |
| 安装方式 | 相机与雷达固定安装，距地面约 2 m，视场重叠 |
| UWB 参考系统 | 五个预标定基站及行人携带的定位标签 |

相机与雷达名义采集频率均为 **10 Hz**，UWB 参考定位频率为 **5 Hz**。实际记录间隔与可用性以时间戳和有效记录为准；各批次的设备配置及坐标参数需分别核对。

### 场景组织

数据按“采集批次与场地 → 工况 → 传感器记录”组织，当前目录名称如下：

| 场景目录 | 工况目录 | 图像数 |
| --- | --- | ---: |
| `0118_playground` | `single_person_normal`、`two_person_normal` | 8,838 |
| `0122_lawn` | `single_person_normal`、`two_person_normal`、`three_person_normal`、`two_person_fog`、`two_person_night` | 31,925 |
| `0129_railway` | `single_person_boundary`、`single_person_fog`、`single_person_night`、`two_person_normal`、`three_person_normal`、`two_person_fog` | 27,101 |

`single_person`、`two_person`、`three_person` 分别表示单人、双人和三人工况；`normal`、`fog`、`night`、`boundary` 分别表示常规、雾天、夜间和边界行走。

`0129_railway/single_person_night/cam/` 的说明指出因光照不足未持续采集可用图像，当前仅保留 10 张 BMP，不应视为完整的多模态配对序列。

### 下载与存储

**数据集下载请联系：[yaoyujiang@hnu.edu.cn](mailto:yaoyujiang@hnu.edu.cn)。** 原始数据不随代码仓库上传。

当前核对的数据包约 **1.02 TB**（十进制，未压缩），主要空间由 BMP 图像占用。下载前请根据所选分包说明预留存储空间，最终下载内容及容量以实际提供的分包为准。

### 目录结构

获取数据后可保留原结构，放入 `dataset/` 或使用外部路径。以下以铁路单人边界工况为例：

```text
dataset/
├── 0118_playground/
├── 0122_lawn/
└── 0129_railway/
    └── single_person_boundary/
        ├── cam/YYYYMMDD_HHMMSS/
        │   └── YYYYMMDD_HHMMSS_mmm.bmp
        ├── radar/
        │   └── LRPD_data_YYYYMMDD_HHMMSS.txt
        └── uwb/
            ├── YYYYMMDD_HHMMSSRTLS1_log.txt
            ├── YYYYMMDD_HHMMSS_RTLS1_Tag<N>_log.csv
            └── YYYYMMDD_HHMMSSRTLS_log_dbg.txt
```

`cam/` 保存图像，`radar/` 保存点云及设备轨迹，`uwb/` 保存参考定位和测距日志。UWB CSV 与调试日志出现在当前铁路批次中，并非每个工况都包含全部类型。同一工况可能包含多次采集及多个日志文件，不能按文件排序或数量直接配对。

### 数据格式

#### 相机图像

图像为未压缩 BMP。对 16 个相机记录目录各自首帧的检查结果均为 **2448 × 2048、24 bit**。文件名为 `YYYYMMDD_HHMMSS_mmm.bmp`，其中 `mmm` 为毫秒部分；采集程序使用电脑本地时钟命名。

#### 雷达日志

TXT 提供处理后的逐帧点云及设备跟踪结果，不包含原始 ADC 波形。帧头形式如下：

```text
Timestamp: YYYY-MM-DD HH:MM:SS.mmm, Frame: <帧号>, Points: <点数>, Tracks: <轨迹数>
```

| 类型 | 字段 |
| --- | --- |
| Points | `X,Y,Z,Doppler,SNR,Noise,TrackIndex` |
| Tracks | `TID,X,Y,Z,Vx,Vy,Vz,Ax,Ay,Az,G,ConfidenceLevel` |

每帧记录数可变；点数和轨迹数均为零时可能只有帧头，读取程序应允许空帧。

#### UWB TXT

主日志包含 `Anchor...` 基站坐标、`TAGn-ANCm=...` 测距及 `TAGn[x,y,z]` 标签位置。不同批次的基站布置可能变化，应使用对应配置解释坐标。日志格式最多保留八个基站字段，实际启用数量以采集配置与有效记录为准。

#### UWB CSV

部分记录按标签提供 CSV，主要字段如下：

| 字段 | 含义或单位 |
| --- | --- |
| `rangetime(ms)` | 毫秒计时值，起点需结合设备说明确认 |
| `tag_ID` | 标签标识 |
| `x(m),y(m),z(m)` | 标签位置，米 |
| `range0(m)` 至 `range7(m)` | 到各基站的距离，米；部分值为 `null` |
| `rx_pwr(dBm)` | 接收功率，dBm |
| `accel_*`、`gyro_*`、`mag_*` | 预留惯性与磁场字段 |
| `roll,pitch,yaw` | 预留姿态字段 |

表头末尾可能存在空列；预留惯性及姿态字段中的零值不能视为有效测量。当前 CSV 文件名中的 `Tag1/Tag2/Tag3` 对应内容中的 `Tag 0/Tag 1/Tag 2`，身份以文件内容及配套映射为准。

### 时间同步与坐标标定

| 数据类型 | 时间形式 |
| --- | --- |
| 相机 | 文件名中的日期、时分秒和毫秒 |
| 雷达 | 帧头完整日期时间 |
| UWB TXT | 日志头日期及各记录的时分秒毫秒 |
| UWB CSV | `rangetime(ms)` |

先明确各时钟的来源、起点、偏移和时区，再进行跨传感器匹配；不能按相同帧号或行号对应，也不能将 `rangetime(ms)` 直接视为 Unix 时间戳。

项目采用相机 **X 向右、Y 向下、Z 向前**，雷达 **X 向右、Y 向前、Z 向上** 的坐标约定；UWB 使用对应批次的基站坐标系，使用前需核对采集配置。雷达投影到图像需要相机内参与雷达–相机三维外参；与 UWB 比较时还需地平面坐标对齐，这两类变换应分别处理。

雷达与 UWB 文本转换参数可通过以下命令查看：

```bash
python -m radar_vision_fusion.preprocessing.parse_radar_lrpd_to_dat --help
python -m radar_vision_fusion.preprocessing.parse_uwb_rtls_to_dat --help
```

视觉入口另需 `image,timestamp` 图像清单（路径相对于清单目录、时间单位为秒）、模型权重和现场标定。

### UWB 参考轨迹与标注

行人携带 UWB 标签提供位置参考；边界参考由携带标签的行人沿预定义边界行走获得。原始 UWB 日志需经过有效性检查、时间对齐、坐标转换和身份关联，才能形成训练或评价参考轨迹；不能直接将原始测量视为无误差真值。

当前原始包未附统一的图像检测框、分割掩码、行人身份、区域顶点或越界事件标注文件。UWB Tag ID、雷达 TID 与图像行人身份不是天然对应关系，多人场景需建立明确映射。

### 论文任务与评价协议

| 任务 | 主要指标 |
| --- | --- |
| 地平面定位与边界重建 | 位置 MAE、RMSE；边界 ASSD、HD95、区域 IoU |
| 轨迹估计 | 带标签 OSPA（labeled OSPA）、位置 RMSE、IDF1、轨迹覆盖率 |
| 行人与区域关系 | 有符号距离误差、区域状态准确率；越界事件 Precision、Recall、F1 及时间误差 |

论文采用区域内部为负、外部为正的有符号距离，边界点计为内部；近边界评价范围为参考距离绝对值不超过 **0.5 m**。越界事件要求转换前后状态各持续至少 **3 帧**，按场景和穿越方向一对一匹配，时间差不超过 **2.0 s**，不要求目标身份一致。时间误差仅针对匹配成功的事件计算，事件指标应结合 IDF1 和轨迹覆盖率解读。

上述为论文评价口径说明；当前公开代码提供基础评测工具，不包含完整的论文实验与事件评价流程。

## 第三方与许可

- [RT-DETR / RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)：[LICENSE](src/third_party/RT-DETR/LICENSE)、[NOTICE](src/third_party/RT-DETR/NOTICE)。
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)：[LICENSE](src/third_party/Depth-Anything-V2/LICENSE)、[NOTICE](src/third_party/Depth-Anything-V2/NOTICE)。

第三方目录保留原许可证和来源，不包含权重或原 Git 历史。**自有代码的许可证、作者署名与论文引用信息待补充。**
