# YOLO Training with CoppeliaSim Dataset

This repository contains the code and instructions to train a basic YOLO model for object detection, using custom images captured from the CoppeliaSim simulator.

The objective is to detect various objects (cubes, spheres, robot parts) on a table for robotics tasks like pick-and-place, with integration into ROS 2 environments.

---

## Project Overview

- **Dataset**: Captured and labeled manually from CoppeliaSim.
- **Framework**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- **Recommended Hardware**: NVIDIA GPU for faster training.
- **Final Goal**: Deploy trained model in CoppeliaSim + ROS 2.

---

## Main Steps

1. **Dataset Creation**  
   Capture and label images with Roboflow or Label Studio.

2. **Setup**  
   Install required packages and prepare data structure.

3. **Training**  
   Fine-tune a YOLO model (e.g., YOLOv11s) on the dataset.

4. **Testing**  
   Evaluate the model on validation images.

5. **Deployment**  
   Integrate the trained model into CoppeliaSim/ROS 2 workflows.

---

## Requirements

- Python 3.8+
- `ultralytics` package (`pip install ultralytics`)
- GPU (optional but highly recommended)

---

## Quick Start

```bash
# Check GPU availability
nvidia-smi

# Install Ultralytics
pip install ultralytics

# Train the model
python train.py
