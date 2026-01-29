# ESP32-S3 Video Streaming Firmware
# ESP32-S3 视频流传输固件

This module is responsible for capturing real-time video data from the OV2640 camera sensor and transmitting JPEG frames via UDP protocol to the host PC.

本模块负责通过 ESP32-S3 驱动 OV2640 摄像头，并将采集到的图像帧通过 UDP 协议实时传输至上位机。

## Hardware (硬件需求)
* **Controller**: ESP32-S3-WROOM-1
* **Camera**:GC2145 (Resolution set to QVGA 320x240 for latency optimization)
* **Protocol**: UDP (User Datagram Protocol)

## Key Features (技术特点)
* **Low Latency**: Uses UDP instead of TCP to prioritize real-time visual feedback for the AI model.
* **Asynchronous Capture**: Implements frame buffering to ensure a stable frame rate (~20-25 FPS).
* **Power Management**: Optimized for battery-powered mobile robot operation.

## How to Flash (如何烧录)
1. Install **Arduino IDE** or **PlatformIO**.
2. Install the ESP32 board support package (v2.0.x recommended).
3. Select board: "ESP32S3 Dev Module".
4. Configure your Wi-Fi SSID and Password in the code.
5. Upload the `.ino` file.

---
**Author**: Kangzhe Zhang (张康哲) 
**Curator**: Kangzhe Zhang (张康哲)
