# rPPG GUI App

A real-time Python application for extracting remote Photoplethysmography (rPPG) signals from webcam input. It features signal plotting, FFT analysis, and face detection — all integrated into a PyQt5 GUI.

## 🧠 Features

- Real-time webcam capture
- Face detection using OpenCV Haar cascades
- Show when the face is detected
- Green channel extraction for rPPG

## TODO

[ ] adding other methods such as CHROM and POS
[ ] Maybe adding TS-CAN ?
[ ] Functionality to select the method before processing begins
[ ] Option to select the face detection algorithm
[ ] Update the costly transfer of images between threads and only send the cropped image

## 📦 Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# Create a virtual environment and install dependencies
uv venv
uv add -r requirements.txt

```
