from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tensorflow as tf


DEFAULT_IMAGE_SIZE = (160, 160)


def load_labels(labels_path: str | Path) -> list[str]:
    """Load class names saved by the training notebook."""
    path = Path(labels_path)
    with path.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"Invalid labels file: {path}")
    return labels


def load_model(model_path: str | Path) -> tf.keras.Model:
    """Load a trained Keras classifier."""
    return tf.keras.models.load_model(model_path)


def preprocess_bgr_frame(
    frame: np.ndarray,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Convert an OpenCV BGR frame into a model-ready batch."""
    if frame is None or frame.size == 0:
        raise ValueError("Cannot preprocess an empty frame")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, image_size, interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    return batch


def predict_frame(
    model: tf.keras.Model,
    labels: Iterable[str],
    frame: np.ndarray,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> tuple[str, float, np.ndarray]:
    """Return the predicted label, confidence, and probability vector."""
    class_names = list(labels)
    batch = preprocess_bgr_frame(frame, image_size=image_size)
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return class_names[index], float(probabilities[index]), probabilities
