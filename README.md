# rPPG GUI App

A real-time Python application for extracting remote Photoplethysmography (rPPG) signals from webcam input. It features signal plotting, FFT analysis, and face detection — all integrated into a PyQt5 GUI.

## 🧠 Features

- Real-time webcam capture
- Face detection using OpenCV Haar cascades
- Green channel extraction for rPPG

## TODO

- adding other methods such as CHROM and POS
- Maybe adding TS-CAN ?
- Functionality to select the method before processing begins

## 📦 Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# Create a virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

```
