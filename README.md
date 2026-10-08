# Real-Time Room Depth Estimation and Human Detection Using YOLOv8

## Abstract
This project combines human detection with monocular depth estimation to estimate the relative depth of people in a room using a standard webcam. It uses YOLOv8 for person detection and a pretrained monocular depth model from MiDaS to generate a depth map. The system displays the live camera view, bounding boxes around detected humans, confidence values, and a depth heatmap in real time.

## Problem statement
In many indoor environments, identifying humans and estimating their spatial position relative to the camera is useful for surveillance, safety monitoring, and interaction systems. Traditional depth sensors are expensive, while monocular depth estimation using a single camera is low-cost and accessible. However, without camera calibration, monocular depth only provides relative depth information rather than exact physical distances.

## Objectives
- Detect humans in real time using YOLOv8.
- Estimate relative room depth with a pretrained monocular depth model.
- Generate a visual depth heatmap.
- Show depth values for each detected human.
- Display live results from a webcam or a still image.
- Support CPU execution as well as CUDA acceleration when available.

## Existing system
Most conventional systems depend on dedicated depth sensors, such as LiDAR or RGB-D cameras, to estimate distance. These systems are expensive, harder to deploy, and may not be practical for everyday room monitoring. Some systems also rely on fixed camera calibration or manual distance mapping.

## Proposed system
The proposed system uses a standard RGB webcam and a pretrained deep learning pipeline to detect people and estimate relative depth from the scene. The application draws bounding boxes, annotates confidence scores, calculates a mean depth value within each detected human region, and overlays the result on the live frame.

## System architecture
The application architecture includes:
1. Input acquisition from webcam or image.
2. Human detection using YOLOv8.
3. Monocular depth estimation using a pretrained MiDaS model.
4. Relative depth extraction for each detected person.
5. Visualization through annotated frames and a heatmap.
6. Optional output saving of captured frames.

## Methodology
The project follows a sequence of steps:
- Capture a frame from the webcam or read an input image.
- Run YOLOv8 to detect person instances.
- Crop each detected person region and compute the mean depth from the depth map.
- Normalize the depth map for visualization.
- Overlay bounding boxes and labels with confidence and relative depth values.
- Display the original frame and depth heatmap in real time.

## Algorithms
- YOLOv8 object detection algorithm for human recognition.
- Monocular depth estimation through a pretrained MiDaS backbone.
- Depth normalization and heatmap generation using OpenCV color mapping.
- Region-based depth aggregation by averaging depth values inside each bounding box.

## Technologies used
- Python
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- NumPy
- Pathlib
- pytest

## Installation
1. Open a terminal in the project folder.
2. Create the virtual environment:
   python -m venv .venv
3. Activate it:
   .\.venv\Scripts\Activate.ps1
4. Upgrade pip:
   python -m pip install --upgrade pip
5. Install dependencies:
   pip install -r requirements.txt

## Execution
Run the webcam version:
python main.py --webcam

Run with a still image:
python main.py --input "path/to/image.jpg"

Controls:
- Q: quit the application
- S: save the current frame

## Expected output
The output window shows:
- Original camera frame with detected humans in green boxes
- Confidence score and relative depth for each person
- A colorized depth heatmap of the room
- Real-time processing of the live scene

## Advantages
- Uses a low-cost webcam instead of a depth sensor
- Works in real time with CPU or GPU support
- Simple and practical for indoor monitoring
- Good for proof-of-concept and research deployments

## Limitations
- Monocular depth is relative and not exact metric depth without camera calibration.
- Depth estimates can be less accurate under poor lighting or ambiguous scenes.
- The model may struggle with unusual camera viewpoints or heavily occluded humans.

## Future scope
- Add camera calibration for metric depth estimation
- Improve depth accuracy with multi-frame fusion
- Extend detection to other objects and scenarios
- Integrate with a web dashboard or cloud backend
- Optimize for edge deployment

## Conclusion
This project demonstrates a practical AI-based system for real-time room depth estimation and person detection using a standard webcam. By combining YOLOv8 and pretrained monocular depth estimation, it provides an accessible and scalable solution for indoor perception tasks. The system estimates relative depth values rather than absolute physical distances unless camera calibration is performed.
