License: MIT (Copyright 2026 Developer6316)
# 🔥 FaceForge

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![OpenCV 4.7](https://img.shields.io/badge/OpenCV-4.7-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Developer6316/Face_Recog/blob/main/colab/FaceForge_Quickstart.ipynb)
[![CI](https://github.com/Developer6316/Face_Recog/actions/workflows/ci.yml/badge.svg)](https://github.com/Developer6316/Face_Recog/actions/workflows/ci.yml)

> **Modular, production-ready face detection and recognition toolkit with GPU acceleration and Google Colab support**

## 🎯 Overview

**FaceForge** is a modular face recognition framework that provides a complete pipeline for face detection, alignment, feature extraction, and identification. Built with flexibility and performance in mind, it supports multiple backends and is optimized for both CPU and GPU environments, including seamless integration with Google Colab.

### Key Features

- **Multiple Detection Methods**: Choose between OpenCV DNN, MTCNN, or RetinaFace detectors.
- **Multiple Recognition Methods**: Leverage FaceNet, InsightFace, or a simple baseline classifier.
- **GPU Acceleration**: Built-in support for CUDA to speed up inference.
- **Google Colab Ready**: One-click setup with a pre-configured Jupyter notebook.
- **Real-time Processing**: Process webcam feeds and video files efficiently.
- **Evaluation Toolkit**: Includes metrics like ROC curves, Precision-Recall, and AUC.
- **Production Ready**: Comes with CI/CD pipelines, Docker support, and unit tests.
- **Extensible**: Easily swap or add new detectors and recognizers.

## 📊 Performance

| Method | Accuracy (LFW) | Speed (FPS) | GPU Support |
| :--- | :--- | :--- | :--- |
| OpenCV DNN | ~93% | 30 | Yes |
| MTCNN | ~95% | 15 | Yes |
| RetinaFace | ~96% | 20 | Yes |
| FaceNet | ~99.6% | 10 | Yes |
| InsightFace | ~99.8% | 15 | Yes |

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

Get started in your browser with no setup required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Developer6316/Face_Recog/blob/main/colab/FaceForge_Quickstart.ipynb)

### Option 2: Local Installation

Follow these steps to set up the project on your local machine.

```bash
# 1. Clone the repository
git clone https://github.com/Developer6316/Face_Recog.git
cd Face_Recog

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download pre-trained models (optional, for full functionality)
bash colab/models_download.sh

# 5. Run the webcam demo
python scripts/demo_webcam.py --device cuda  # Use --device cpu if no GPU

# 6. Process a video file
python scripts/process_video.py path/to/input.mp4 --output detections.jsonl

# 7. Run the test suite
pytest -q
```
# Docker Support
## Deploy the application inside a container with GPU support.
```bash
# Build the Docker image
docker build -f docker/Dockerfile.gpu -t faceforge .

# Run the container with GPU and webcam access
docker run --gpus all -it --rm \
  --device /dev/video0:/dev/video0 \
  faceforge
```

# 📁 Project Structure
```text
Face_Recog/
├── .github/workflows/        # CI/CD pipelines (GitHub Actions)
│   └── ci.yml
├── colab/                    # Google Colab resources
│   ├── FaceForge_Quickstart.ipynb
│   └── models_download.sh
├── docker/                   # Docker configuration files
│   ├── Dockerfile.gpu
│   └── docker-compose.yml
├── docs/                     # Documentation
│   ├── architecture.md
│   └── usage.md
├── examples/                 # Example images/videos
├── scripts/                  # Command-line utilities
│   ├── demo_webcam.py
│   ├── process_video.py
│   └── evaluate_dataset.py
├── src/faceforge/            # Core Python package
│   ├── detectors/            # Face detection backends
│   │   ├── opencv_dnn.py
│   │   ├── mtcnn.py
│   │   └── retinaface.py
│   ├── recognizers/          # Face recognition backends
│   │   ├── facenet.py
│   │   ├── insightface.py
│   │   └── sv_classifier.py
│   ├── utils/                # Utility functions
│   │   ├── video.py
│   │   ├── transforms.py
│   │   ├── gpu.py
│   │   └── metrics.py
│   ├── api.py                # High-level FaceForge pipeline
│   └── cli.py                # Command-line interface
├── tests/                    # Unit and integration tests
├── .gitignore
├── LICENSE                   # MIT License
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.cfg                 # Package configuration
└── pyproject.toml            # Modern Python packaging
```
## 🔧 Usage Examples
Basic Pipeline
```python
from faceforge import FaceForge
from faceforge.detectors import OpenCVDNNDetector
from faceforge.recognizers import SimpleVerifier

# 1. Initialize the pipeline
detector = OpenCVDNNDetector()
recognizer = SimpleVerifier()
pipeline = FaceForge(detector, recognizer)

# 2. Process an image (BGR format)
image = cv2.imread('path/to/image.jpg')
results = pipeline.process_image(image)

# 3. Examine results
for result in results:
    box = result['box']                 # [x1, y1, x2, y2]
    score = result['score']             # Detection confidence
    embedding = result['embedding']     # Feature vector
    identity = result['identity']       # Identified person (if enrolled)

    print(f"Detected: {identity} with confidence {score:.2f}")
Enrolling New Faces
python
from faceforge.recognizers import FaceNetRecognizer

# Load the FaceNet recognizer
recognizer = FaceNetRecognizer()

# Enroll a new person
for name, face_images in your_dataset.items():
    # face_images: list of cropped and aligned face images
    embedding = recognizer.embed(face_images[0])  # Use average of multiple embeddings
    recognizer.enroll(name, embedding)

# Now the recognizer can identify the person
identity = recognizer.identify(new_face_embedding)
print(f"Identified as: {identity}")
```
# 📈 Evaluation
Run the evaluation script to benchmark your model's performance.

```bash
python scripts/evaluate_dataset.py --preds predictions.jsonl
Sample Output:

AUC: 0.9876
Accuracy: 95.4%
Precision: 0.96
Recall: 0.95
🧪 Testing
We use pytest for testing. Ensure all dependencies are installed.

bash
# Run all tests
pytest -v

# Run with coverage report
pytest --cov=src/faceforge

# Run a specific test file
pytest tests/test_detectors.py
```
# 🐳 Docker Support
## For GPU-accelerated inference inside a container:

```bash
# Build the image
docker build -f docker/Dockerfile.gpu -t faceforge:gpu .

# Run with GPU access
docker run --gpus all -it --rm faceforge:gpu --webcam
For CPU-only environments, use a standard Python base image.
```
🤝 Contributing
We welcome contributions! Please follow these steps:

Fork the repository.

Create a new branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.

Please ensure your code adheres to our coding standards and includes appropriate tests.

📝 License
Distributed under the MIT License. See LICENSE for more information.

🙏 Acknowledgments
PyTorch for the deep learning framework.

OpenCV for computer vision utilities.

FaceNet and InsightFace for face recognition models.

MTCNN for face detection.

📧 Contact
Developer6316 - GitHub

Project Link: https://github.com/Developer6316/Face_Recog

⭐ Star History
If you find this project useful, please consider giving it a star! Your support is highly appreciated.

https://api.star-history.com/svg?repos=Developer6316/Face_Recog&type=Date
