# Autonomous-Driving-Robot-System
A full-stack autonomous robot system integrating ESP32-S3 vision, PC-end Deep Learning (YOLOv8 &amp; CNN), and Pyboard-based PID hardware control.

# Autonomous Driving Robot System: A Full-Stack Perspective
# 基于深度学习与硬件闭环的自动驾驶小车系统

##  Overview (项目简介)
This project implements an end-to-end autonomous driving robot platform. It integrates high-level computer vision (YOLOv8 & CNN) with low-level precise hardware control (Pyboard & ESP32-S3). The system is designed to demonstrate the full cycle of autonomous navigation, including perception, wireless low-latency communication, and sensor-fusion-based execution.

本项目实现了一个端到端的自动驾驶机器人平台。它结合了上位机计算机视觉（YOLOv8 与 CNN）与底层精准硬件控制（Pyboard 与 ESP32-S3），展示了从感知、低延迟通信到基于传感器融合的执行全过程。

---

##  System Architecture (系统架构)
The system is divided into three layers:
1. **Perception Layer**: ESP32-S3 camera captures CIF resolution frames and streams via a customized **UDP protocol** (EnhancedUDPGetter) to ensure latency < 100ms.
2. **Decision Layer**: PC-end Python environment running **YOLOv8** for human following and **SimpleCNN** for autonomous lane keeping.
3. **Execution Layer**: MicroPython-based Pyboard processing commands via Bluetooth, integrating **MPU6050 (IMU)** and **Ultrasonic sensors** for closed-loop motion control.

---

## Academic & Technical Highlights (学术与技术亮点)

### 1. Temperature-Compensated IMU Fusion (传感器温漂修正)
* **Challenge**: MPU6050 gyroscope exhibits significant Z-axis drift due to temperature changes.
* **Solution**: Established a regression model between angular velocity and temperature using **Python**.
* **Result**: Reduced heading angle error to **within 1°**, enabling precise PID-based differential steering.

### 2. Real-time Vision Pipelines (实时视觉处理)
* **YOLOv8 Human Following**: Real-time object detection with adaptive distance control based on bounding box area.
* **SimpleCNN Lane Keeping**: A lightweight convolutional neural network trained on customized datasets for robust lane tracking.

### 3. Communication Strategy (通信策略)
* Implemented **UDP fragmentation and reassembly** to prioritize "Real-time over Completeness," a core principle in autonomous driving to prevent command accumulation.

---

##  Repository Structure (仓库结构)
* `/Software/PC_Inference`: Python code for YOLOv8 and CNN inference.
* `/Software/Firmware`: MicroPython scripts for Pyboard (PID, MPU6050 driver, Ultrasonic).
* `/Hardware`: Circuit schematics and component list (BOM).
* `/Dataset`: Sample images used for training the SimpleCNN model.

---

##  Performance (性能表现)
| Metric (指标) | Value (数值) |
| :--- | :--- |
| End-to-End Latency (端到端延迟) | < 100 ms |
| Heading Error (航向角误差) | < 1.0° / min |
| Detection FPS (检测帧率) | 30+ FPS (PC-side) |

---

##  Demo (效果演示)
*【【开源】基于Pyboard1.1.CN全栈自研自动驾驶小车：YOLOv8 视觉决策与 IMU 温漂补偿融合系统】 https://www.bilibili.com/video/BV1nCziBTEzr/?share_source=copy_web&vd_source=bea9fb3d3d926a0b3135c058968748ab*

---

##  License
Distributed under the **MIT License**. See `LICENSE` for more information.
