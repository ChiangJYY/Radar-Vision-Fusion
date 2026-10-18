# -*- coding: utf-8 -*-

import sys
import platform
import os
import threading
import time
from datetime import datetime
from ctypes import *
import numpy
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
import argparse

# # 兼容不同操作系统加载动态库
# currentsystem = platform.system()
# if currentsystem == 'Windows':
#     sys.path.append(os.path.join(os.getenv('MVCAM_COMMON_RUNENV'), "Samples", "Python", "MvImport"))
# else:
#     sys.path.append(os.path.join("..", "..", "MvImport"))
#

# 根据你的SDK安装位置添加路径
sdk_path = r"D:\MVS\MVS\Development\Samples\Python\MvImport"
# 或者
# sdk_path = r"C:\Program Files\MindVision\Runtime\Samples\Python\MvImport"

if os.path.exists(sdk_path):
    sys.path.append(sdk_path)
    # print(f"已添加SDK路径: {sdk_path}")
else:
    print(f"错误: 路径不存在 - {sdk_path}")
    print("请确认SDK是否正确安装")

# 现在再导入模块
from MvCameraControl_class import *

# 兼容Python 2.x和3.x的输入处理
if sys.version_info[0] < 3:
    input_func = raw_input
else:
    input_func = input


def IsHBPixelFormat(enPixelType=0):
    """判断是否为HB格式像素"""
    if enPixelType in (PixelType_Gvsp_HB_Mono8,
                       PixelType_Gvsp_HB_Mono10,
                       PixelType_Gvsp_HB_Mono10_Packed,
                       PixelType_Gvsp_HB_Mono12,
                       PixelType_Gvsp_HB_Mono12_Packed,
                       PixelType_Gvsp_HB_Mono16,
                       PixelType_Gvsp_HB_RGB8_Packed,
                       PixelType_Gvsp_HB_BGR8_Packed,
                       PixelType_Gvsp_HB_RGBA8_Packed,
                       PixelType_Gvsp_HB_BGRA8_Packed,
                       PixelType_Gvsp_HB_RGB16_Packed,
                       PixelType_Gvsp_HB_BGR16_Packed,
                       PixelType_Gvsp_HB_RGBA16_Packed,
                       PixelType_Gvsp_HB_BGRA16_Packed,
                       PixelType_Gvsp_HB_YUV422_Packed,
                       PixelType_Gvsp_HB_YUV422_YUYV_Packed,
                       PixelType_Gvsp_HB_BayerGR8,
                       PixelType_Gvsp_HB_BayerRG8,
                       PixelType_Gvsp_HB_BayerGB8,
                       PixelType_Gvsp_HB_BayerBG8,
                       PixelType_Gvsp_HB_BayerRBGG8,
                       PixelType_Gvsp_HB_BayerGB10,
                       PixelType_Gvsp_HB_BayerGB10_Packed,
                       PixelType_Gvsp_HB_BayerBG10,
                       PixelType_Gvsp_HB_BayerBG10_Packed,
                       PixelType_Gvsp_HB_BayerRG10,
                       PixelType_Gvsp_HB_BayerRG10_Packed,
                       PixelType_Gvsp_HB_BayerGR10,
                       PixelType_Gvsp_HB_BayerGR10_Packed,
                       PixelType_Gvsp_HB_BayerGB12,
                       PixelType_Gvsp_HB_BayerGB12_Packed,
                       PixelType_Gvsp_HB_BayerBG12,
                       PixelType_Gvsp_HB_BayerBG12_Packed,
                       PixelType_Gvsp_HB_BayerRG12,
                       PixelType_Gvsp_HB_BayerRG12_Packed,
                       PixelType_Gvsp_HB_BayerGR12,
                       PixelType_Gvsp_HB_BayerGR12_Packed):
        return True
    else:
        return False


def IsMonoPixelFormat(enPixelType=0):
    """判断是否为单色格式像素"""
    if enPixelType in (PixelType_Gvsp_Mono8,
                       PixelType_Gvsp_Mono10,
                       PixelType_Gvsp_Mono10_Packed,
                       PixelType_Gvsp_Mono12,
                       PixelType_Gvsp_Mono12_Packed,
                       PixelType_Gvsp_Mono14,
                       PixelType_Gvsp_Mono16):
        return True
    else:
        return False


def decoding_char(ctypes_char_array):
    """安全地从ctypes字符数组中解码出字符串"""
    byte_str = memoryview(ctypes_char_array).tobytes()
    null_index = byte_str.find(b'\x00')
    if null_index != -1:
        byte_str = byte_str[:null_index]
    for encoding in ['gbk', 'utf-8', 'latin-1']:
        try:
            return byte_str.decode(encoding)
        except UnicodeDecodeError:
            continue
    return byte_str.decode('latin-1', errors='replace')


class ImageSignal(QObject):
    """用于线程间通信的信号类"""
    image_ready = pyqtSignal(numpy.ndarray, int, str)  # 图像数据, 帧号, 时间戳


class CameraCaptureGUI(QMainWindow):
#设置帧率
    def __init__(self):
        super().__init__()
        self.cam = None
        self.deviceList = None
        self.is_grabbing = False
        self.is_saving = False
        self.save_dir = None
        self.frame_count = 0
        self.start_time = None
        self.timer = None
        self.callback_func = None  # 保持回调函数引用
        # 取消在构造时固定帧率，由外部通过 set_target_fps 设置
        self.target_fps = None
        self.exposure_ms = None

        # 图像信号
        self.image_signal = ImageSignal()
        self.image_signal.image_ready.connect(self.update_image)

        # 初始化SDK
        MvCamera.MV_CC_Initialize()

        self.init_ui()
        self.enum_devices()

    def set_camera_resolution(self, width, height):
        """设置相机分辨率"""
        try:
            # 设置宽度
            ret = self.cam.MV_CC_SetIntValue("Width", width)
            if ret != 0:
                print(f"设置宽度失败! ret[0x{ret:x}]")
                return

            # 设置高度
            ret = self.cam.MV_CC_SetIntValue("Height", height)
            if ret != 0:
                print(f"设置高度失败! ret[0x{ret:x}]")
                return

            print(f"分辨率已设置为 {width}x{height}")
        except Exception as e:
            print(f"设置分辨率时出错: {str(e)}")
#设置最大曝光时间
    def set_auto_exposure_upper_limit(self, upper_limit_ms=25):
        """Set the upper limit for auto exposure time."""
        try:
            # Convert milliseconds to microseconds (as required by the SDK)
            upper_limit_us = upper_limit_ms * 1000

            # Set the upper limit for auto exposure time
            ret = self.cam.MV_CC_SetIntValue("AutoExposureTimeUpperLimit", upper_limit_us)
            if ret != 0:
                print(f"Failed to set Auto Exposure Upper Limit! ret[0x{ret:x}]")
                return

            print(f"Auto Exposure Upper Limit set to {upper_limit_ms} ms")
        except Exception as e:
            print(f"Error setting Auto Exposure Upper Limit: {str(e)}")

    def set_manual_exposure_time(self, exposure_ms):
        """设置手动曝光时间(毫秒)。会同时关闭自动曝光。"""
        if self.cam is None:
            return False

        try:
            if exposure_ms is None:
                self.exposure_ms = None
                return False
            exposure_ms = float(exposure_ms)
            if exposure_ms <= 0:
                return False
            self.exposure_ms = exposure_ms

            # 先关闭自动曝光
            exposure_auto_off = False
            for param_name in ["ExposureAuto", "ExposureAutoMode"]:
                ret = self.cam.MV_CC_SetEnumValue(param_name, 0)  # 0=Off
                if ret == 0:
                    exposure_auto_off = True
                    break
            if not exposure_auto_off:
                print("警告: 关闭自动曝光失败，可能不支持此功能")

            # SDK常见单位为微秒(us)
            exposure_us = exposure_ms * 1000.0

            # 尝试常见曝光时间参数名
            last_ret = None
            for param_name in ["ExposureTime", "ExposureTimeAbs"]:
                ret = self.cam.MV_CC_SetFloatValue(param_name, exposure_us)
                if ret == 0:
                    print(f"手动曝光已设置为 {exposure_ms:.3f} ms ({param_name}={exposure_us:.1f} us)")
                    return True
                last_ret = ret

                # 有些设备是整数接口
                ret2 = self.cam.MV_CC_SetIntValue(param_name, int(exposure_us))
                if ret2 == 0:
                    print(f"手动曝光已设置为 {exposure_ms:.3f} ms ({param_name}={int(exposure_us)} us)")
                    return True
                last_ret = ret2

            print(f"设置手动曝光失败! ret[0x{int(last_ret):x}]" if last_ret is not None else "设置手动曝光失败")
            return False
        except Exception as e:
            print(f"设置手动曝光时出错: {str(e)}")
            return False

    def _on_exposure_spin_changed(self, value):
        """UI曝光时间变更回调"""
        self.set_manual_exposure_time(value)

    def set_exposure_ms(self, exposure_ms):
        """在主线程更新曝光显示，并尝试应用到已连接相机。"""
        try:
            if exposure_ms is None:
                self.exposure_ms = None
            else:
                self.exposure_ms = float(exposure_ms)
        except Exception as e:
            print("设置曝光失败，非法数值: %s" % str(e))
            return

        if hasattr(self, "spin_exposure") and self.spin_exposure is not None and self.exposure_ms is not None:
            QTimer.singleShot(0, lambda: self.spin_exposure.setValue(self.exposure_ms))

        if self.cam is not None and self.exposure_ms is not None:
            self.set_manual_exposure_time(self.exposure_ms)

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("摄像头采集系统")
        self.setGeometry(100, 100, 1200, 800)

        # 主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # 控制面板
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.StyledPanel)
        control_layout = QHBoxLayout()
        control_frame.setLayout(control_layout)

        self.btn_start = QPushButton("开始")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.clicked.connect(self.start_capture)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_capture)

        # 信息显示标签
        self.label_frame_count = QLabel("总帧数: 0")
        self.label_timestamp = QLabel("时间戳: --")
        # 显示当前目标帧率
        self.label_fps = QLabel("FPS: --")
        self.label_elapsed_time = QLabel("计时: 00:00:00")

        # 曝光时间控制（手动）
        from PyQt5.QtWidgets import QDoubleSpinBox
        self.label_exposure = QLabel("曝光(ms):")
        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setDecimals(3)
        self.spin_exposure.setRange(0.001, 1000.0)
        self.spin_exposure.setSingleStep(0.5)
        self.spin_exposure.setValue(50.0)
        self.spin_exposure.valueChanged.connect(self._on_exposure_spin_changed)

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addStretch()
        control_layout.addWidget(self.label_frame_count)
        control_layout.addWidget(self.label_timestamp)
        control_layout.addWidget(self.label_fps)
        control_layout.addWidget(self.label_exposure)
        control_layout.addWidget(self.spin_exposure)
        control_layout.addWidget(self.label_elapsed_time)

        # 图像显示区域
        self.label_image = QLabel()
        self.label_image.setMinimumSize(800, 600)
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setText("等待图像...")
        self.label_image.setStyleSheet("border: 1px solid gray; background-color: black; color: white;")
        self.label_image.setScaledContents(False)  # 不自动缩放，保持比例

        main_layout.addWidget(control_frame)
        main_layout.addWidget(self.label_image, 1)

        # 定时器用于更新计时
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_elapsed_time)

    def enum_devices(self):
        """枚举设备"""
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE
                      | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)

        ret = MvCamera.MV_CC_EnumDevices(tlayerType, self.deviceList)
        if ret != 0:
            self.show_error("枚举设备失败! ret[0x%x]" % ret)
            return

        if self.deviceList.nDeviceNum == 0:
            self.show_error("未找到设备!")
            return

        print("找到 %d 个设备!" % self.deviceList.nDeviceNum)

        # 自动选择第一个设备
        if self.deviceList.nDeviceNum > 0:
            self.connect_camera(0)

    def connect_camera(self, device_index):
        """连接相机"""
        try:
            # 创建相机实例
            self.cam = MvCamera()

            # 选择设备并创建句柄
            stDeviceList = cast(self.deviceList.pDeviceInfo[device_index], POINTER(MV_CC_DEVICE_INFO)).contents

            ret = self.cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                self.show_error("创建句柄失败! ret[0x%x]" % ret)
                return

            # 打开设备
            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                self.show_error("打开设备失败! ret[0x%x]" % ret)
                return

            # 探测网络最佳包大小(只对GigE相机有效)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE or stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE:
                nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                    if ret != 0:
                        print("警告: 设置包大小失败! ret[0x%x]" % ret)

            # 设置触发模式为off
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                print("警告: 设置触发模式失败! ret[0x%x]" % ret)

            # # 设置采集帧率（如果设备支持）
            # if self.target_fps and self.target_fps > 0:
            #     ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
            #     if ret != 0:
            #         print("警告: 开启帧率控制失败! ret[0x%x]" % ret)
            #     else:
            #         ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", self.target_fps)
            #         if ret != 0:
            #             print("警告: 设置帧率失败! ret[0x%x]" % ret)
            #         else:
            #             print("帧率已设置为 %.2f fps" % self.target_fps)

            # 关闭自动曝光，改为手动曝光
            exposure_auto_off = False
            for param_name in ["ExposureAuto", "ExposureAutoMode"]:
                ret = self.cam.MV_CC_SetEnumValue(param_name, 0)  # 0=Off
                if ret == 0:
                    print(f"自动曝光已关闭 ({param_name})")
                    exposure_auto_off = True
                    break
            if not exposure_auto_off:
                print("警告: 关闭自动曝光失败，可能不支持此功能")

            # 设置自动增益模式
            # MVS SDK枚举值: 0=Off, 1=Once, 2=Continuous
            # 尝试多种可能的参数名
            gain_auto_set = False
            for param_name in ["GainAuto", "GainAutoMode"]:
                # 使用整数枚举值: 2 表示 Continuous (连续自动增益)
                ret = self.cam.MV_CC_SetEnumValue(param_name, 2)
                if ret == 0:
                    print(f"自动增益模式已设置为连续模式 ({param_name})")
                    gain_auto_set = True
                    break
            if not gain_auto_set:
                print("警告: 设置自动增益模式失败，可能不支持此功能")


            print("相机连接成功!")

            # 连接成功后，如果UI里已有曝光值，则尝试应用手动曝光
            try:
                if hasattr(self, "spin_exposure") and self.spin_exposure is not None:
                    self.set_manual_exposure_time(self.spin_exposure.value())
            except Exception:
                pass

        except Exception as e:
            self.show_error("连接相机时出错: %s" % str(e))

    def start_capture(self):
        """开始采集或恢复保存"""
        if self.cam is None:
            self.show_error("相机未连接!")
            return

        # 如果已经在采集，只是恢复保存
        if self.is_grabbing:
            if not self.is_saving:
                # 恢复保存，创建新的保存目录
                self.start_time = datetime.now()
                timestamp_str = self.start_time.strftime("%Y%m%d_%H%M%S")
                self.save_dir = timestamp_str
                os.makedirs(self.save_dir, exist_ok=True)
                self.frame_count = 0  # 重新开始计数
                self.is_saving = True

                # 更新UI
                self.btn_start.setEnabled(False)
                self.btn_stop.setEnabled(True)
                self.timer.start(100)  # 重新开始计时

                # print("恢复保存图像...")
            return

        # 首次开始采集
        # 创建保存目录（保存在当前工作目录下）
        self.start_time = datetime.now()
        timestamp_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = timestamp_str
        os.makedirs(self.save_dir, exist_ok=True)

        self.is_grabbing = True
        self.is_saving = True
        self.frame_count = 0

        # 注册回调函数
        fun_ctype = get_platform_functype()
        FrameInfoCallBack2 = fun_ctype(None, POINTER(MV_FRAME_OUT), c_void_p, c_bool)
        self.callback_func = FrameInfoCallBack2(self.image_callback)  # 保持引用

        ret = self.cam.MV_CC_RegisterImageCallBackEx2(self.callback_func, ctypes.py_object(self), True)
        if ret != 0:
            self.show_error("注册回调失败! ret[0x%x]" % ret)
            self.is_grabbing = False
            return

        # 开始取流
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.show_error("开始取流失败! ret[0x%x]" % ret)
            self.is_grabbing = False
            return

        # 更新UI
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.timer.start(100)  # 每100ms更新一次计时

        print("开始采集图像...")

    def stop_capture(self):
        """停止保存（但继续显示画面）"""
        if not self.is_grabbing:
            return

        # 只停止保存，不停止取流和显示
        self.is_saving = False

        # 更新UI
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.timer.stop()

        # print("停止保存图像. 共保存 %d 帧，画面继续显示" % self.frame_count)

    def image_callback(self, pstFrame, pUser, bAutoFree):
        """图像回调函数"""
        if not self.is_grabbing:
            return

        try:
            stFrame = cast(pstFrame, POINTER(MV_FRAME_OUT)).contents
            if stFrame is None:
                return

            # 处理图像
            numpy_image = self.process_frame(stFrame)
            if numpy_image is not None:
                # 生成时间戳
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒

                # 保存图像
                if self.is_saving:
                    self.save_image(numpy_image, timestamp)

                # 发送信号更新UI
                self.image_signal.image_ready.emit(numpy_image, stFrame.stFrameInfo.nFrameNum, timestamp)

            # 非自动释放模式，需要手动释放资源
            if bAutoFree == False:
                if pUser is not None:
                    User = ctypes.cast(pUser, ctypes.py_object).value
                    User.MV_CC_FreeImageBuffer(stFrame)

        except Exception as e:
            print("回调函数出错: %s" % str(e))

    def process_frame(self, stFrame):
        """处理图像帧"""
        try:
            stDecodeParam = MV_CC_HB_DECODE_PARAM()
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))

            # 如果是HB图像，需要进行HB解码
            result = IsHBPixelFormat(stFrame.stFrameInfo.enPixelType)
            if result:
                DecodeBufferlen = stFrame.stFrameInfo.nWidth * stFrame.stFrameInfo.nHeight * 3
                DecodeBuffer = (c_ubyte * DecodeBufferlen)()

                stDecodeParam.pSrcBuf = stFrame.pBufAddr
                stDecodeParam.nSrcLen = stFrame.stFrameInfo.nFrameLen
                stDecodeParam.pDstBuf = DecodeBuffer
                stDecodeParam.nDstBufSize = DecodeBufferlen
                ret = self.cam.MV_CC_HBDecode(stDecodeParam)
                if ret != 0:
                    print("HB解码失败! ret[0x%x]" % ret)
                    return None

                stConvertParam.pSrcData = stDecodeParam.pDstBuf
                stConvertParam.nSrcDataLen = stDecodeParam.nDstBufLen
                stConvertParam.enSrcPixelType = stDecodeParam.enDstPixelType
            else:
                stConvertParam.pSrcData = stFrame.pBufAddr
                stConvertParam.nSrcDataLen = stFrame.stFrameInfo.nFrameLen
                stConvertParam.enSrcPixelType = stFrame.stFrameInfo.enPixelType

            # 图像格式转换
            enDstPixelType = PixelType_Gvsp_Undefined
            nChannelNum = 1
            bMono = IsMonoPixelFormat(stConvertParam.enSrcPixelType)
            if bMono:
                enDstPixelType = PixelType_Gvsp_Mono8
                nChannelNum = 1
            else:
                enDstPixelType = PixelType_Gvsp_RGB8_Packed
                nChannelNum = 3

            convertdestBufflen = nChannelNum * stFrame.stFrameInfo.nWidth * stFrame.stFrameInfo.nHeight
            DstBuffer = (c_ubyte * convertdestBufflen)()

            stConvertParam.nWidth = stFrame.stFrameInfo.nWidth
            stConvertParam.nHeight = stFrame.stFrameInfo.nHeight
            stConvertParam.enDstPixelType = enDstPixelType
            stConvertParam.pDstBuffer = DstBuffer
            stConvertParam.nDstBufferSize = convertdestBufflen

            ret = self.cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
            if ret != 0:
                print("像素转换失败! ret[0x%x]" % ret)
                return None

            # 转换为numpy数组
            if nChannelNum == 1:
                numpy_image = numpy.frombuffer(DstBuffer, dtype=numpy.ubyte, count=convertdestBufflen). \
                    reshape(stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth)
            else:
                numpy_image = numpy.frombuffer(DstBuffer, dtype=numpy.ubyte, count=convertdestBufflen). \
                    reshape(stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth, 3)
                # RGB转BGR用于OpenCV显示
                numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

            return numpy_image

        except Exception as e:
            print("处理帧时出错: %s" % str(e))
            return None

    def save_image(self, numpy_image, timestamp):
        """保存图像"""
        try:
            filename = "%s.bmp" % timestamp
            filepath = os.path.join(self.save_dir, filename)
            cv2.imwrite(filepath, numpy_image)
            self.frame_count += 1
        except Exception as e:
            print("保存图像失败: %s" % str(e))

    def update_image(self, numpy_image, frame_num, timestamp):
        """更新显示的图像"""
        try:
            # 调整图像大小以适应显示区域
            label_size = self.label_image.size()
            h, w = numpy_image.shape[:2]

            # 计算缩放比例，保持宽高比
            scale_w = label_size.width() / w if label_size.width() > 0 else 1.0
            scale_h = label_size.height() / h if label_size.height() > 0 else 1.0
            scale = min(scale_w, scale_h)  # 使用较小的比例以保持宽高比

            # 只有当标签大小有效时才缩放
            if label_size.width() > 0 and label_size.height() > 0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                if new_w > 0 and new_h > 0:
                    display_image = cv2.resize(numpy_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    display_image = numpy_image
            else:
                display_image = numpy_image

            # 转换为QImage
            if len(display_image.shape) == 2:  # 灰度图
                q_image = QImage(display_image.data, display_image.shape[1], display_image.shape[0],
                                 display_image.strides[0], QImage.Format_Grayscale8)
            else:  # 彩色图
                q_image = QImage(display_image.data, display_image.shape[1], display_image.shape[0],
                                 display_image.strides[0], QImage.Format_BGR888)

            # 显示图像
            pixmap = QPixmap.fromImage(q_image)
            self.label_image.setPixmap(pixmap)

            # 更新信息显示
            self.label_frame_count.setText("总帧数: %d" % self.frame_count)
            self.label_timestamp.setText("时间戳: %s" % timestamp)

        except Exception as e:
            print("更新图像时出错: %s" % str(e))

    def update_elapsed_time(self):
        """更新已用时间"""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            hours = elapsed.seconds // 3600
            minutes = (elapsed.seconds % 3600) // 60
            seconds = elapsed.seconds % 60
            self.label_elapsed_time.setText("计时: %02d:%02d:%02d" % (hours, minutes, seconds))

    def show_error(self, message):
        """显示错误信息"""
        print("错误: %s" % message)
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "错误", message)

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止保存
        self.is_saving = False

        # 停止取流
        if self.is_grabbing and self.cam:
            try:
                self.cam.MV_CC_StopGrabbing()
            except:
                pass

        self.is_grabbing = False
        self.timer.stop()

        # 关闭相机
        if self.cam:
            try:
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()
            except:
                pass

        # 反初始化SDK
        try:
            MvCamera.MV_CC_Finalize()
        except:
            pass

        event.accept()

    def set_target_fps(self, fps):
        """在主线程更新目标帧率显示，并尝试应用到已连接的相机"""
        try:
            if fps is None:
                self.target_fps = None
            else:
                self.target_fps = float(fps)
        except Exception as e:
            print("设置帧率失败，非法数值: %s" % str(e))
            return

        # 更新UI标签（确保在主线程执行）
        display_text = "FPS: --" if (self.target_fps is None) else ("FPS: %.2f" % self.target_fps)
        QTimer.singleShot(0, lambda: self.label_fps.setText(display_text))

        # 如果相机已连接，则尝试应用帧率设置
        if self.cam is not None:
            if self.target_fps and self.target_fps > 0:
                ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
                if ret != 0:
                    print("警告: 开启帧率控制失败! ret[0x%x]" % ret)
                else:
                    ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", self.target_fps)
                    if ret != 0:
                        print("警告: 设置帧率失败! ret[0x%x]" % ret)
                    else:
                        print("帧率已设置为 %.2f fps" % self.target_fps)
            else:
                # 关闭帧率控制（如果设备支持）
                ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)
                if ret != 0:
                    print("警告: 关闭帧率控制失败! ret[0x%x]" % ret)


if __name__ == "__main__":
    # 支持命令行参数以便在脚本末尾直接修改参数，也可直接编辑下面的默认值
    parser = argparse.ArgumentParser(description="Camera capture parameters")
    parser.add_argument("--fps", type=float, default=10, help="Target frames per second")
    parser.add_argument("--width", type=int, default=2448, help="Resolution width")
    parser.add_argument("--height", type=int, default=2048, help="Resolution height")
    parser.add_argument("--exposure-ms", type=float, default=50, help="Manual exposure time (ms)")
    args = parser.parse_args()

    # 可直接在这里修改默认值，或通过命令行覆盖，例如:
    # python camera2.py --fps 25 --width 1280 --height 720 --exposure-ms 40
    DEFAULT_FPS = args.fps
    DEFAULT_RESOLUTION = (args.width, args.height)
    DEFAULT_EXPOSURE_MS = args.exposure_ms

    app = QApplication(sys.argv)
    window = CameraCaptureGUI()
    # 应用配置（通过 set_target_fps 以更新 UI 并尝试应用到相机）
    window.set_target_fps(DEFAULT_FPS)
    window.set_camera_resolution(*DEFAULT_RESOLUTION)
    window.set_exposure_ms(DEFAULT_EXPOSURE_MS)

    window.show()
    sys.exit(app.exec_())

