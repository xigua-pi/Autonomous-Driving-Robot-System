# Core Inference & Decision Logic

This directory contains the primary decision-making scripts that process real-time visual data and output hardware control commands.

##  Modules

### 1. `include_yolo8_simpleCNN.py` (Hybrid Mode)
The flagship controller of the system. It features a dual-model switching logic:
* **Cruise Mode**: Uses a custom **SimpleCNN** to regress steering angles for autonomous lane keeping.
* **Follow Mode**: Switches to **YOLOv8** when a specific target is detected, enabling high-level interaction.

### 2. `only_yolo8.py` (Target Tracking Mode)
A dedicated tracking module focused on human-robot interaction.
* **Safety Mechanism (Fail-safe)**: Implements a strict "Detection-to-Action" interlock. If the model fails to identify a "person" within the current frame (confidence thresholding), the system triggers an **immediate emergency stop (S)**. 
* **Design Philosophy**: Prioritizes safety over continuity, preventing the robot from moving blindly in uncertain environments.

---

##  Key Implementation Details

### Fail-safe Logic (安全保护机制)
In `only_yolo8.py`, we implemented a robust stopping logic:
```python
 # 丢失目标或未发现目标
                state['last_target_center'] = None
                state['target_box'] = None

                # 修改部分：不再使用 CNN 巡航，直接停止
                speed_char = "S"
                steer_val = 0.0
                state['mode_text'] = "MODE: IDLE (NO TARGET)"
