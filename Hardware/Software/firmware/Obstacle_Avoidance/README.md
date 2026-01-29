# Firmware: Pyboard-based Autonomous Obstacle Avoidance
# 底层固件：基于 Pyboard 的自主避障系统

This directory contains the core firmware designed to run directly on the **Pyboard** microcontroller inside the robot. These scripts manage the critical low-latency tasks: sensor fusion, motor driving, and the hardware-level safety interlock.

本项目底层固件运行于小车内部的 **Pyboard** 微控制器。该模块负责处理高实时性任务，包括传感器融合、电机驱动以及硬件级安全保护。

---

## Logic & Module Breakdown (逻辑与模块划分)

The following scripts are deployed on the Pyboard to coordinate the robot's physical movement:

* **`main.py`**: The system's main entry point on Pyboard. It coordinates the initialization of peripherals and executes the primary control loop.
* **`robot_mind.py`**: The "Intelligence" layer of the firmware. It processes distance data and makes the final decision on whether to follow AI commands or initiate emergency avoidance.
* **`ultrasonic.py`**: A dedicated driver for HC-SR05 sensors. It uses high-precision timers to measure distances without blocking the CPU's other tasks.
* **`ir_receive.py`**: Infrared signal processing, allowing the robot to be manually toggled between "AI Cruise" and "Manual Override" modes via a remote.

---

## Safety Architecture (安全架构)

The firmware serves as the **Fail-safe Layer**. Even if the top-level PC (running YOLOv8/CNN) experiences latency or disconnection, the Pyboard maintains local control:

1.  **Continuous Polling**: `ultrasonic.py` scans the environment every 50ms.
2.  **Autonomous Override**: If `robot_mind.py` detects an obstacle within the danger zone, it immediately overrides the Serial/Bluetooth commands from the PC to prevent a collision.



---

## Hardware Specification (硬件规格)

* **Processor**: STM32F405RG (Pyboard v1.1)
* **Language**: MicroPython
* **Sensor Array**: 3-channel Ultrasonic sensors (Front/Left/Right)
* **Actuators**: DC Motors controlled via Dual-channel PWM
* **Motor Driver**:L298N

---

## Contribution (项目贡献)

* **Author (代码作者)**: Xianjun Deng (邓贤君)
* **Curator (内容整理)**: Kangzhe Zhang (张康哲)

---
[Back to Project Main Menu](../../README.md)
