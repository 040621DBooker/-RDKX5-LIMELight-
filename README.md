# User Guide for Bridge Underwater Rebar Exposure Detection System Based on RDK X5 and LIMELight Low-Light Enhancement

## 📌 Project Overview

To address the issues of low efficiency, high cost, and significant safety risks in underwater structural health monitoring for rebar exposure, this project targets underwater bridge inspection scenarios. It proposes a fully automated solution integrating low-light image enhancement and intelligent detection. The improved LIMELight algorithm applies multi-region block correction to effectively handle uneven brightness and scattering interference in underwater images. Combined with an optimized YOLOv11 detection model incorporating cross-scale feature fusion and attention mechanisms, the system enables precise detection of underwater rebar exposure. The complete solution is deployed on the RDK X5 edge computing platform, ensuring accurate on-device detection.

---

## 📂 File Structure

This project provides the following core files and program code components:

- `main.cc`: Main program file, includes image input, enhancement, inference, and visualization;
- `LIME.bin`: Quantized rebar detection model executable on RDK X5;
- `IMG_6005_RESIZE_LIME.mp4`: Simulated underwater rebar video for demo;
- `CMakeLists.txt`: C++ build script for the RDK board.

---

## 🚀 Getting Started

### 1. Environment Setup

Please follow the official documentation to set up the runtime environment on the RDK X5 board.

---

### 2. Compile the Program

Run the following commands in the source directory:

```bash
mkdir build && cd build
cmake ..
make -j8
```

---

### 3. Modify Configuration

Update the following macro definitions in `main.cc` according to your actual file paths:

```cpp
#define MODEL_PATH "/path/to/LIME.bin"
#define VIDEO_PATH "/path/to/input_video.mp4"
#define OUTPUT_VIDEO_PATH "cpp_result_video.avi"
```

---

### 4. Run the Program

Ensure the paths to the video and model are correct, then run:

```bash
./main
```

The program will automatically read video frames, perform image enhancement and object detection, display results in real-time, and output the result video to the specified location.

---

## ⚠️ Notes

### 🔒 About LIMELight Algorithm

The **LIMELight** image enhancement algorithm is an original contribution from our team, specifically optimized for underwater low-light scenarios with uneven illumination and color distortion.

> **Important Note**: As the related paper is currently under review, to preserve academic originality, **the core module is not open-sourced at this time**.

For collaboration, please contact us using the information below.

---

### 📹 Video and Model Requirements

- Recommended video resolution: 640×640, frame rate ≥ 10 FPS;
- Use Horizon Open Explorer toolchain to convert YOLOv11 ONNX model to `.bin` format;
- Only INT8 post-training quantized (PTQ) models are supported for inference.

---

## ✅ System Performance

Based on tests on the RDK X5 platform, this system achieves:

- Image enhancement time: ~60 ms per frame;
- Object detection time: ~15 ms per frame;
- Overall frame rate: stable at 13 FPS;
- Suitable for real-time underwater video stream processing.

---

## 🧩 Expandability

This system supports the following extensions:

- Replace the detection model (requires re-quantization);
- Switch input source to camera or live stream;
- Integrate with path planning for underwater robotics;
- Modular enhancement/detection functions for multi-class object recognition.

---

## 📞 Contact Us

For licensing, model cooperation, or deployment support, please contact our team:

- Project Name: RDK X5 Underwater Rebar Exposure Detection System
- Team Name: Jin Dao’s Underwater Steel
- Email: 1208779078@qq.com

---

> © This project is a preliminary result. Some components are not open source and are provided for demo and integration testing only. Redistribution without permission is prohibited.
