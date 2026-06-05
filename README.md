# 🎯 Object Detection and Tracking

> **CodeAlpha Internship — Artificial Intelligence Track | Task 4**

Real-time object detection and tracking using YOLOv8 and ByteTrack.
Supports webcam and video file input with live FPS display, motion trails, and screenshot capture.

## Features
- ⚡ Real-time detection using YOLOv8n
- 🎯 Multi-object tracking with ByteTrack (unique ID per object)
- 🌈 Color-coded bounding boxes per object class
- 🐾 Motion trails showing object movement history
- 📊 Live FPS and object count HUD
- 💾 Optional video output saving
- 📸 Screenshot capture with S key

## Tech Stack
| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv8 model + ByteTrack |
| `opencv-python` | Video I/O and frame rendering |

## Getting Started

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_ObjectDetectionTracking.git
cd CodeAlpha_ObjectDetectionTracking
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run
```bash
# Webcam
python3 detect_track.py

# Video file
python3 detect_track.py --source path/to/video.mp4

# Save output
python3 detect_track.py --save
```

## Project Structure
CodeAlpha_ObjectDetectionTracking/
├── detect_track.py
├── utils/
│   ├── init.py
│   ├── draw.py
│   └── tracker_config.yaml
├── requirements.txt
└── README.md

## Controls
| Key | Action |
|---|---|
| Q | Quit |
| S | Save screenshot |

## License
Developed as part of the CodeAlpha AI Internship Program.