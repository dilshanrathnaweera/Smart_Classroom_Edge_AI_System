# Smart Classroom Edge AI System

An Edge AI-powered system that detects classroom occupancy from video, performs real-time inference on an edge device, automatically simulates air conditioning control, and provides live monitoring through a web dashboard.

**Course:** Edge Computing | **Instructor:** Thulasee Shan | **University of Jaffna, Class of 2026**

---

## Overview

This project implements an end-to-end Edge AI pipeline for smart classroom management. It classifies classroom occupancy into **LOW**, **MEDIUM**, and **HIGH** levels from video footage, runs inference locally on an edge device, and drives an automated air conditioning response based on the detected occupancy — all visualized through a live Streamlit dashboard.

## Key Features

- **Occupancy Detection** — Classifies real classroom video into LOW / MEDIUM / HIGH occupancy levels
- **Edge Inference** — Model runs locally on the edge device; no cloud dependency during inference
- **Automated AC Simulation** — AC state (OFF / 24°C / 20°C) responds automatically to live occupancy
- **Live Dashboard** — Real-time monitoring of occupancy, AC state, temperature, and runtime
- **Containerized Deployment** — Fully packaged with Docker for consistent, portable deployment
- **MLOps Workflow** — Supports retraining, model export, and redeployment as new data is collected

## Architecture

```
Capture → Train (Cloud) → Package (Docker) → Run on Edge → Act & Display
```

Video is captured and used to train a classification model in the cloud. The trained model is exported, containerized with Docker, and deployed to run inference locally on the edge device. Detected occupancy drives the simulated AC and is surfaced live on the dashboard.

## Project Structure

```text
Smart_Classroom_Edge_AI_System/
├── dashboard/         # Streamlit web dashboard (app, utils, requirements)
├── docker/            # Dockerfile and container configuration
├── model/             # Trained occupancy detection model, labels, and evaluation scripts
├── dataset/           # Dataset documentation and sample classroom footage
├── documentation/     # Project documentation
├── .gitignore
├── LICENSE
└── README.md
```

## Dataset

Sample classroom footage and full dataset documentation are available in [`dataset/`](./dataset). The complete raw video dataset used for training exceeds GitHub's practical storage limits, so it is hosted externally on Google Drive — the access link is provided in the `dataset/` folder's documentation. This repository includes only sample clips and dataset metadata; no raw training footage is stored in version control.

## How setup and run in our pc?

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Build and Run

```bash
cd docker
docker build -t smart-classroom-edge-ai .
docker run -d -p 8501:8501 smart-classroom-edge-ai
```

Then open **http://localhost:8501** in your browser to access the dashboard.                                                                                       (For better Result set Detection Confidence Threshold value to 0.15)

### Stopping the Container

```bash
docker stop <container_id>
```

## Team Structure

| Role | Responsibilities |
|---|---|
| **Product Owner** | Requirement gathering, system architecture, final demonstration |
| **Project Manager / Scrum Master** | Team coordination, progress tracking, repository management, presentation management |
| **App Developers** | Edge application, dashboard development, AC simulation |
| **Data Scientists** | Dataset collection, model training, model deployment and MLOps |

## Privacy & Ethics

All training footage excludes visible faces and participants under the age of 15, in accordance with dataset collection guidelines. Consent was secured from all individuals appearing in recorded footage.

## License

This project is licensed under the terms specified in [LICENSE](./LICENSE).
