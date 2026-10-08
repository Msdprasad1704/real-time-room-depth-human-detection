# Real-Time Room Depth Estimation and Human Detection

A real-time Deep Learning and Computer Vision application that detects humans and estimates relative room depth using a standard webcam.

The system combines **YOLOv8** for human detection and **MiDaS** for monocular depth estimation. It provides real-time visualization through bounding boxes, depth information, heatmaps, person tracking, screenshots, video recording, and CSV reports.

---

## 📌 Project Overview

The system uses a standard RGB webcam to understand the indoor environment.

It performs two major tasks:

1. Detect humans in the camera frame using YOLOv8.
2. Estimate relative depth using the MiDaS monocular depth-estimation model.

The detected humans are displayed with bounding boxes, confidence scores, tracking information, and relative depth values.

A colorized depth heatmap is also generated to visualize the approximate depth structure of the scene.

> **Note:** MiDaS provides relative depth estimation. It does not directly provide exact physical distance in meters unless the system is calibrated.

---

## 🎯 Objectives

- Detect humans in real time using YOLOv8.
- Estimate relative room depth using MiDaS.
- Generate a real-time depth heatmap.
- Calculate relative depth for detected people.
- Track detected people.
- Display confidence and depth information.
- Support webcam and image input.
- Generate screenshots.
- Record processed video.
- Generate CSV reports.
- Support CPU and CUDA-enabled GPU execution when available.

---

## ✨ Key Features

### 👤 Human Detection

Uses YOLOv8 to detect people in real time.

### 📏 Relative Depth Estimation

Uses the pretrained MiDaS model to estimate the relative depth of objects and people.

### 🌡️ Depth Heatmap

Generates a colorized heatmap representing the relative depth structure of the scene.

### 🆔 Person Tracking

Detected people can be tracked across video frames.

### 📊 Depth Information

Displays relative depth information for detected people.

### 📸 Screenshot Capture

Press `S` to save the current processed frame.

Screenshots are stored in:
## 📸 Sample Output

### Real-Time Human Detection and Depth Estimation

![Real-Time Depth Estimation Output](assets/sample-output.png)


The system detects humans using YOLOv8 and generates a relative depth map using MiDaS.

```text
outputs/screenshots/
