# Smart Classroom Edge AI – Data Science Module

This repository contains the **Data Science module** of the **Smart Classroom Edge AI System**. It focuses on building the computer vision model that detects and counts classroom occupants. The trained model provides occupancy information that is used by the main application to simulate intelligent air conditioner (AC) control.

> **Note:** This repository contains only the data science components. The frontend, backend, and IoT simulation are maintained in the main project repository.

---

# Objective

The objective of this module is to develop a lightweight object detection model capable of:

* Detecting people in classroom images
* Classifying them as **Student** or **Janitor**
* Counting only students for occupancy estimation
* Exporting the trained model for edge deployment

---

# Data Science Pipeline

```text
Classroom Video
        │
        ▼
Frame Extraction
        │
        ▼
Image Annotation
        │
        ▼
Dataset Preparation
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
ONNX Export
        │
        ▼
Edge Inference
        │
        ▼
Student Count
```

---

# Dataset Preparation

## Data Collection

* Classroom videos were recorded under different classroom conditions.
* Frames were extracted using **Python** and **OpenCV**.
* Images were captured at **1 frame per second**.
* Original image resolutions (720p and 1080p) were preserved to maintain detection quality.

---

## Image Annotation

Image annotation was performed using **Azure Custom Vision**.

### Object Classes

* Student
* Janitor

Bounding boxes were manually drawn around every visible person in the dataset.

The dataset includes:

* Different classroom layouts
* Various lighting conditions
* Different viewing angles
* Small and large crowds
* Partial occlusions

---

# Model Training

The object detection model was trained using:

* Azure Custom Vision
* General Compact Domain
* YOLO-based architecture
* Input resolution: **512 × 512**

The General Compact domain was selected because it supports exporting the trained model to the ONNX format for edge deployment.

---

# Model Export

After training, the model was exported as:

* `model.onnx`
* `labels.txt`

Inference is performed using **ONNX Runtime**, enabling efficient execution on CPU-based edge devices without requiring a dedicated GPU.

---

# Inference Pipeline

```text
Input Image
      │
      ▼
ONNX Runtime
      │
      ▼
Object Detection
      │
      ▼
Student Count
      │
      ▼
Occupancy Level
```

---

# Occupancy Logic

Only **Student** detections contribute to occupancy.

| Student Count | Occupancy Level |
| ------------- | --------------- |
| 0 – 2         | Low             |
| 3 – 9         | Medium          |
| 10+           | High            |

Janitor-only detections are ignored when calculating occupancy.

---

# Technologies Used

* Python
* OpenCV
* Azure Custom Vision
* ONNX Runtime
* YOLO-based Object Detection

---

# Repository Structure

```text
data-science/
│
├── dataset/
│   ├── images/
│   └── annotations/
│
├── models/
│   ├── model.onnx
│   └── labels.txt
│
├── scripts/
│   ├── extract_frames.py
│   ├── train_model.md
│   ├── inference.py
│   └── occupancy_logic.py
│
├── notebooks/
│
├── README.md
└── requirements.txt
```




