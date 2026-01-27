# Perception & Decision Engine (AI Module)

This module handles the high-level intelligence of the robot, including real-time video stream processing, object detection (YOLOv8), and autonomous lane following (CNN).

##  Code Map (代码映射)

### 1. Real-time Inference (核心实时推理)
* `include_yolo8_simpleCNN.py`: **The Master Controller**. Integrates both YOLOv8 (for target following) and SimpleCNN (for cruise) with a state-machine logic.
* `only_yolo8.py`: Pure YOLOv8 mode for robust human/object tracking.


### 2. Training Pipeline (模型训练全流程)
* `model.py`: Definition of the **SimpleCNN architecture** (a 3-layer ConvNet optimized for embedded-to-PC inference).
* `dataset.py`: Custom PyTorch Dataset class for loading and augmenting robot-eye perspective data.
* `train.py` & `train_cpu.py`: Training scripts using **SmoothL1Loss** (Huber Loss) to achieve precise steering angle regression.

### 3. Data Engineering (数据工程)
* `collect_data.py`: A specialized tool to sync UDP video frames with manual control labels for dataset creation.
* `yolo8_check.py`: Environment diagnostic tool to ensure CUDA/GPU acceleration is active for YOLOv8.

##  Technical Innovation: EnhancedUDPGetter
One of the key contributions in this module is the `EnhancedUDPGetter` class (found in `basic_run_model.py`). 
* **The Problem**: Standard TCP/IP streaming causes "Head-of-Line Blocking," leading to 2-3 second lags.
* **The Solution**: A multi-threaded UDP receiver that implements a **"Latest-Frame-First"** buffer. By dropping stale packets, we maintain a consistent **80-100ms** glass-to-glass latency, which is critical for high-speed autonomous driving.

---
**Author**: Kangzhe Zhang (张康哲)
