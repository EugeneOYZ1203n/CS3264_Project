# CS3264 Final Project — Singapore Sign Language (SgSL) Recognition

> **Course:** CS3264 — Foundations of Machine Learning  
> **Final Project Direction:** Real-time Sign Language Recognition for Singapore Sign Language (SgSL)

## 🎬 Demo

[![SgSL Demo](https://img.youtube.com/vi/H-ZxMfkzUIM/maxresdefault.jpg)](https://youtu.be/H-ZxMfkzUIM)

▶️ [Watch the demo on YouTube](https://youtu.be/H-ZxMfkzUIM)

---

## Overview

This project develops a real-time **Sign Language Recognition (SLR)** system for **Singapore Sign Language (SgSL)** — a relatively low-resource sign language used by the Deaf community in Singapore.

Given the scarcity of labelled SgSL data, our approach leverages:
- **Pose-based landmark extraction** via MediaPipe Holistic (hands + body keypoints) rather than raw video, keeping the model lightweight and signer-agnostic
- **Transfer learning / fine-tuning** on SgSL data after pre-training on larger sign language datasets (e.g. Google ASL)
- A **Transformer-based sequence model** that ingests normalised keypoint sequences and classifies them into SgSL signs

The system runs in real-time from a video file and displays a live prediction dashboard.

---

## Repository Structure

```
CS3264_Project/
├── pose_extract/               # Scripts to extract MediaPipe pose landmarks from raw videos
├── slr/                        # Core model training, preprocessing and inference code
    ├── model.py                # SLRModel definition (Transformer-based)
    ├── preprocess.py           # Keypoint normalisation, interpolation, POSE_INDICES
    ├── data_augmentation.py    # Data augmentation for poses
    ├── train.py                # Training loop (OneCycle LR, AWP, multi-phase)
    └── sgsl/
        └── label_map.json      # SgSL class → index mapping

```

---

## Model

- **Architecture:** Transformer encoder over per-frame keypoint feature vectors
- **Input:** 134-dimensional feature vector per frame (21 left hand + 21 right hand + 25 selected pose keypoints, each as x/y coordinates → flattened)
- **Output:** Softmax over SgSL sign classes
- **Training strategy:**
  - Pre-trained / fine-tuned on Google ASL dataset (`feature_extract=True` phase then full fine-tune)
  - Data augmentation 
  - OneCycle learning rate scheduler
  - Adversarial Weight Perturbation (AWP) for regularisation
  - Multi-phase training curriculum

---

## Preprocessing

Implemented in `preprocess.py`:

- **Landmark selection:** 25 body keypoints from MediaPipe's 33-point pose model (upper body + arms), defined by `POSE_INDICES`
- **Missing value interpolation:** Linear interpolation across time for occluded/undetected keypoints
- **Normalisation:** Global min-max normalisation per sequence to make predictions signer- and camera-agnostic

---

## Acknowledgements

- [MediaPipe](https://mediapipe.dev/) for real-time holistic landmark detection
- [Google ASL Dataset](https://www.kaggle.com/competitions/asl-signs) used for pre-training
- SgSL data collected and labelled as part of this project