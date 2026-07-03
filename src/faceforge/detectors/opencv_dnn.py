from typing import List, Dict
import numpy as np
import cv2
import os
import logging

logger = logging.getLogger(__name__)

class Detector:
    def detect(self, image: np.ndarray) -> List[Dict]:
        raise NotImplementedError

class OpenCVDNNDetector(Detector):
    """Simple OpenCV DNN face detector wrapper using a Caffe/Tensorflow model if available.
    Falls back to Haar cascades when no model found.
    """
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = device
        self.net = None
        self.cascade = None
        
        if model_path and os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNet(model_path)
                if device.startswith("cuda"):
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception as e:
                logger.warning(f"Failed to load DNN model: {e}. Falling back to Haar Cascade")
                self.net = None
        
        # Try to load Haar cascade if available
        if self.net is None:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.cascade = cv2.CascadeClassifier(cascade_path)
            except Exception as e:
                logger.warning(f"Failed to load Haar Cascade: {e}. Face detection will not work properly.")
                self.cascade = None

    def detect(self, image: np.ndarray) -> List[Dict]:
        h, w = image.shape[:2]
        
        if self.net:
            try:
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.net.setInput(blob)
                detections = self.net.forward()
                results = []
                for i in range(detections.shape[2]):
                    conf = float(detections[0, 0, i, 2])
                    if conf > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype("int")
                        results.append({"box": [x1, y1, x2, y2], "score": conf})
                return results
            except Exception as e:
                logger.warning(f"DNN detection failed: {e}. Falling back to cascade if available.")
                if self.cascade is None:
                    return []
        
        # Fallback to Haar cascade
        if self.cascade is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rects = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                results = []
                for (x, y, w_, h_) in rects:
                    results.append({"box": [x, y, x + w_, y + h_], "score": 1.0})
                return results
            except Exception as e:
                logger.warning(f"Cascade detection failed: {e}")
                return []
        
        # No detector available
        logger.warning("No face detector available (neither DNN nor Cascade)")
        return []
