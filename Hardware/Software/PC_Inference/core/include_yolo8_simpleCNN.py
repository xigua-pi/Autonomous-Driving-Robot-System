import cv2  # 引入opencv
import torch  # 引入pytorch
import numpy as np  # 矩阵运算库
import threading  # 多线程库
import socket  # 网络库，用于通过UDP协议接收摄像头数据
import time  # 时间库，用于控制速度和计算延迟
import serial  # 串口库，用于和蓝牙模块通信发送指令
from collections import deque  # 队列库，用于存放最近几次的历史数据（平滑处理）
from PIL import Image  # 图像处理库，用于转换图片格式供AI使用
from torchvision import transforms  # AI预处理工具，用于缩放图片、转张量
from model import SimpleCNN  # 从本地文件 model.py 导入实验人训练的simpleCNN结构
from ultralytics import YOLO  # 导入YOLOv8目标检测模型

# ---配置区---
UDP_PORT = 1234  # 接收视频数据的端口号
BT_COM_PORT = 'COM9'  # 蓝牙模块在实验者电脑上的串口号（需根据实际修改）
BT_BAUD_RATE = 9600  # 蓝牙通信波特率，HC-05默认9600
MODEL_WEIGHTS = "model.pth"  # CNN巡航模型的权重文件名
YOLO_MODEL_PATH = "best.pt"  # 修改为yolo8的模型文件名

# 滤波与控制参数
SMOOTH_WINDOW_SIZE = 5  # 滑动平均窗口大小，值越大控制越丝滑但反应越慢
DEBOUNCE_THRESHOLD = 2  # 去抖动阈值，防止指令频繁跳变
DEAD_ZONE = 0.18  # 转向死区，偏移量小于此值则视为直行
CMD_INTERVAL = 0.1  # 向小车发送指令的时间间隔（单位：秒）

# 速度与距离配置
TARGET_AREA_RATIO = 0.60  # 目标理想面积占比 (0~1)
AREA_TOLERANCE = 0.10  # 面积容差，在此范围内不前后移动
YOLO_CONF_THRES = 0.35  # YOLO检测的置信度阈值，越高越准确但容易漏检

#---配置区结束---

# 负责通过UDP持续接收视频帧的类
class EnhancedUDPGetter:
    def __init__(self, port=UDP_PORT):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建UDP套接字
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 允许地址重用
        self.sock.bind(("0.0.0.0", self.port))  # 绑定端口，监听所有IP
        self.frame = None  # 存储最新的一帧画面
        self.stopped = False  # 控制线程停止的标志
        self.buffer = bytearray()  # 字节缓冲区，用于拼接图片数据
        self.last_frame_time = time.time()  # 记录上一帧收到的时间

        # 新增统计变量 
        self.packet_count = 0  # 累计收到的包数
        self.frame_count = 0  # 成功解码的帧数
        self.start_time = time.time()

    def start(self):
        # 启动一个后台线程专门接收数据，不阻塞主程序
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                data, _ = self.sock.recvfrom(65535)  # 接收原始UDP数据包
                self.packet_count += 1  # 统计数据包

                start_idx = data.find(b'\xff\xd8')  # 寻找JPEG图片的开头标识
                if start_idx != -1:
                    self.buffer = bytearray(data[start_idx:])  # 如果是开头，重置缓冲区
                else:
                    self.buffer.extend(data)  # 否则持续拼接数据

                if b'\xff\xd9' in data:  # 寻找JPEG图片的结尾标识
                    end_pos = self.buffer.find(b'\xff\xd9')
                    if end_pos != -1:
                        img_data = self.buffer[:end_pos + 2]  # 截取完整的图片字节流
                        nparr = np.frombuffer(img_data, np.uint8)  # 转为numpy数组
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 解码成OpenCV图像
                        if img is not None:
                            self.frame = img  # 更新当前画面
                            self.frame_count += 1  # 统计成功解码的帧数
                            self.last_frame_time = time.time()  # 更新时间戳
                        self.buffer = bytearray()  # 清空缓冲区
            except Exception as e:
                if not self.stopped:
                    print(f"UDP Error: {e}")
                continue

    def stop(self):
        self.stopped = True  # 停止标志位
        try:
            self.sock.close()  # 关闭网络连接
        except:
            pass


# 负责将AI产生的原始值转换为平滑控制指令的类
class ControlSmoother:
    def __init__(self, window_size=5, debounce_frames=3):
        self.history = deque(maxlen=window_size)  # 存储最近几帧的历史记录
        self.debounce_frames = debounce_frames  # 切换指令所需的连续帧数
        self.last_stable_cmd = "S"  # 上一次发送的稳定指令
        self.counter = 0  # 计数器，用于指令去抖

    def smooth(self, steer_val, speed_cmd):
        """
        steer_val: 转向值 (-1~1)
        speed_cmd: 速度字符 ('F', 'B', 'S')
        """
        # 判定转向方向 
        if steer_val > DEAD_ZONE:
            steer_dir = "R"  # 右转
        elif steer_val < -DEAD_ZONE:
            steer_dir = "L"  # 左转
        else:
            steer_dir = "F"  # 直行 (Forward)

        # 组合指令，例如 'FF' (前进+直行), 'FR' (前进+右转), 'BF' (后退+直行)
        # 如果速度是停止 'S'，则指令就是 'S'
        current_combined = speed_cmd if speed_cmd == "S" else f"{speed_cmd}{steer_dir}"

        # 去抖动逻辑 
        if current_combined == self.last_stable_cmd:
            self.counter = 0  # 如果指令没变，计数器清零
            return current_combined
        else:
            self.counter += 1  # 如果指令变了，开始计数
            if self.counter >= self.debounce_frames:  # 只有连续多次一样，才更新稳定指令
                self.last_stable_cmd = current_combined
                self.counter = 0
                return current_combined
            return self.last_stable_cmd  # 否则维持旧指令


# 主运行函数
def run_inference():
    # 检测电脑是否有显卡，没有则用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 硬件连接初始化
    try:
        bt = serial.Serial(BT_COM_PORT, BT_BAUD_RATE, timeout=0.1)  # 尝试打开串口
        print(f"[OK] Bluetooth on {BT_COM_PORT}")
    except:
        bt = None  # 如果打开失败，进入仿真模式（不发串口指令）
        print("[WARN] Simulation Mode")

    # 2. 加载AI模型
    cnn_model = SimpleCNN().to(device)  # 创建CNN模型实例并移至设备
    cnn_model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))  # 加载权重文件
    cnn_model.eval()  # 设置为预测模式
    yolo_model = YOLO(YOLO_MODEL_PATH)  # 加载YOLO模型

    # 3. 初始化状态变量（字典形式方便跨线程读写）
    state = {
        'current_cmd': "S",  # 当前生成的最终指令
        'mode_text': "IDLE",  # 界面显示的模式文字
        'target_box': None,  # 当前锁定的目标框坐标
        'last_target_center': None,  # 上一帧目标的中心点（用于锁定目标）
        'running': True,  # 程序运行状态
        'latency': 0  # AI推理耗时
    }

    getter = EnhancedUDPGetter().start()  # 启动视频接收器
    smoother = ControlSmoother(SMOOTH_WINDOW_SIZE, DEBOUNCE_THRESHOLD)  # 初始化平滑器
    # 定义图片进入CNN前的预处理流程：缩放为64x64，转为PyTorch张量
    preprocess = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    #  AI 推理工作线程 
    def inference_worker():
        nonlocal state  # 声明使用外部的state变量
        while state['running']:
            frame = getter.frame  # 获取视频接收器中的最新一帧
            if frame is None:
                time.sleep(0.01)  # 如果没画面，小睡一下继续等
                continue

            t_start = time.time()  # 记录开始处理的时间
            h, w, _ = frame.shape  # 获取画面高和宽

            # 使用YOLO检测人（classes=0），设置较小尺寸imgsz=320以保证画面的延迟较低
            yolo_results = yolo_model.predict(frame, classes=0, conf=YOLO_CONF_THRES, verbose=False, device=device,
                                              imgsz=320)

            steer_val = 0.0  # 初始转向值
            speed_char = "S"  # 初始速度状态

            # 目标锁定与跟随逻辑
            best_box = None
            if len(yolo_results[0].boxes) > 0:  # 如果画面里检测到了人
                boxes = yolo_results[0].boxes
                if state['last_target_center'] is None:
                    # 第一次发现，选置信度最高的第一个人
                    best_box = boxes[0]
                else:
                    # 寻找中心点距离上一次最近的目标，防止切换到背景中的其他人
                    min_dist = float('inf')
                    for b in boxes:
                        curr_center = b.xywh[0][:2].cpu().numpy()  # 获取当前候选人的中心坐标
                        dist = np.linalg.norm(curr_center - state['last_target_center'])  # 计算欧式距离
                        if dist < min_dist:
                            min_dist = dist
                            best_box = b

                if best_box:
                    # 更新锁定目标的中心点
                    xywh = best_box.xywh[0].cpu().numpy()  # [中心x, 中心y, 宽, 高]
                    state['last_target_center'] = xywh[:2]  # 记忆中心点
                    state['target_box'] = best_box.xyxy[0].cpu().numpy()  # 记忆框坐标

                    # 计算水平偏离 (转向值范围 -1 到 1)
                    steer_val = (xywh[0] - (w / 2)) / (w / 2)

                    # 基于面积的自适应速度 ---
                    box_area_ratio = (xywh[2] * xywh[3]) / (w * h)  # 计算目标占画面的面积比
                    if box_area_ratio < (TARGET_AREA_RATIO - AREA_TOLERANCE):
                        speed_char = "F"  # 离得远 (面积小)，前进追赶
                    elif box_area_ratio > (TARGET_AREA_RATIO + AREA_TOLERANCE):
                        speed_char = "B"  # 离得太近 (面积大)，后退
                    else:
                        speed_char = "S"  # 距离刚好，停止移动（仅保持转向）

                    state['mode_text'] = "MODE: LOCK-FOLLOW"  # 切换文字为跟随模式
            else:
                # 丢失目标
                state['last_target_center'] = None
                state['target_box'] = None

                # 切换到 CNN 巡航模式（看路行驶）
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转为RGB格式
                input_tensor = preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(device)  # 预处理
                with torch.no_grad():  # 不计算梯度，节省内存
                    steer_val = cnn_model(input_tensor).item()  # CNN 预测转向值
                speed_char = "F"  # 巡航模式默认前进
                state['mode_text'] = "MODE: CNN-CRUISE"

            # 将预测的转向和速度交给平滑器，更新最终指令
            state['current_cmd'] = smoother.smooth(steer_val, speed_char)
            state['latency'] = (time.time() - t_start) * 1000  # 计算AI总延迟

    # 开启AI线程
    threading.Thread(target=inference_worker, daemon=True).start()

    # 4. 主循环 (负责：发送串口指令、窗口画面渲染)
    last_bt_time = 0  # 记录上一次发指令的时间
    last_stat_time = time.time()

    # 统计平滑变量 
    avg_net_latency = 0.0
    ema_alpha = 0.1  # 指数移动平均系数，值越小越平滑
    loss_rate = 0.0
    fps = 0
    prev_frame_count = 0
    prev_packet_count = 0

    try:
        while True:
            frame = getter.frame
            if frame is None: continue

            h, w, _ = frame.shape  # 在主循环中定义 h 和 w
            now = time.time()

            # --- 网络状态计算 (带平滑滤波) ---
            raw_latency = (now - getter.last_frame_time) * 1000
            # 使用 EMA 滤波防止数值剧烈跳动
            avg_net_latency = (ema_alpha * raw_latency) + (1 - ema_alpha) * avg_net_latency

            # 每秒更新一次丢包率和FPS
            if now - last_stat_time > 1.0:
                elapsed = now - last_stat_time
                current_frames = getter.frame_count - prev_frame_count
                current_packets = getter.packet_count - prev_packet_count

                fps = int(current_frames / elapsed)
                # 简单丢包计算：如果预期每帧1个包（实际上JPEG可能分包，这里作为相对参考）
                if current_packets > 0:
                    loss_rate = max(0, (1 - (current_frames / (current_packets if current_packets > 0 else 1))) * 100)

                prev_frame_count = getter.frame_count
                prev_packet_count = getter.packet_count
                last_stat_time = now

            # 限制发送频率
            if bt and (now - last_bt_time > CMD_INTERVAL):
                if (now - getter.last_frame_time) > 0.5:
                    bt.write(b'S')
                else:
                    bt.write(state['current_cmd'].encode())
                last_bt_time = now

            #  渲染可视化画面 
            display_frame = frame.copy()
            if state['target_box'] is not None:
                b = state['target_box'].astype(int)
                cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cv2.circle(display_frame, tuple(state['last_target_center'].astype(int)), 5, (0, 0, 255), -1)

            # 绘制延迟和网络状态
            stat_line1 = f"Avg Net: {avg_net_latency:.1f}ms | AI: {state['latency']:.1f}ms"
            stat_line2 = f"FPS: {fps} | Loss: {loss_rate:.1f}% | Total: {getter.packet_count}"

            cv2.putText(display_frame, stat_line1, (10, h - 45), 1, 1.0, (0, 255, 255), 1)
            cv2.putText(display_frame, stat_line2, (10, h - 20), 1, 1.0, (0, 255, 255), 1)

            # 在画面左上角叠加模式文字和当前指令
            cv2.putText(display_frame, f"{state['mode_text']} | CMD: {state['current_cmd']}", (10, 30), 1, 1.2,
                        (255, 0, 0), 2)

            # 显示画面窗口
            cv2.imshow("Smart Control System", display_frame)

            # 检测键盘按下 'q' 退出程序
            if cv2.waitKey(1) & 0xFF == ord('q'):
                state['running'] = False
                break
    finally:
        # 程序收尾工作
        state['running'] = False
        getter.stop()  # 停止网络接收
        if bt:
            bt.write(b'S')  # 退出前务必让小车停下
            bt.close()  # 关闭串口
        cv2.destroyAllWindows()  # 关闭所有显示窗口


# 程序入口
if __name__ == "__main__":
    run_inference()
