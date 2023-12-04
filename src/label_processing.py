from pathlib import Path

import cv2
import numpy as np

from src.colors import BLUE, GREEN, RED, WHITE, Color
from src.constants import LABEL_CODE_TO_NAME, OVERLAP_THRESHOLD


class CV2FormatLabel:
    def __init__(
        self,
        normalized_min_x: float,
        normalized_min_y: float,
        normalized_max_x: float,
        normalized_max_y: float,
        label_code: int,
    ):
        self.normalized_min_x = normalized_min_x
        self.normalized_min_y = normalized_min_y
        self.normalized_max_x = normalized_max_x
        self.normalized_max_y = normalized_max_y
        self.label_code = int(label_code)

    def normalized_area(self) -> float:
        return (self.normalized_max_x - self.normalized_min_x) * (
            self.normalized_max_y - self.normalized_min_y
        )


def labels_overlap(label1: CV2FormatLabel, label2: CV2FormatLabel) -> bool:
    labels = label1, label2
    x_intersection = max(
        0.0,
        min([label.normalized_max_x for label in labels])
        - max([l.normalized_min_x for l in labels]),
    )
    y_intersection = max(
        0.0,
        min([label.normalized_max_y for label in labels])
        - max([l.normalized_min_y for l in labels]),
    )
    intersection = x_intersection * y_intersection
    union = label1.normalized_area() + label2.normalized_area() - intersection
    iou = intersection / union
    return iou > OVERLAP_THRESHOLD


def labels_yolo_to_cv2_format(yolo_format_labels: np.ndarray) -> list[CV2FormatLabel]:
    labels = yolo_format_labels[:, 0]
    min_xs = yolo_format_labels[:, 1] - yolo_format_labels[:, 3] / 2
    min_ys = yolo_format_labels[:, 2] - yolo_format_labels[:, 4] / 2
    max_xs = yolo_format_labels[:, 1] + yolo_format_labels[:, 3] / 2
    max_ys = yolo_format_labels[:, 2] + yolo_format_labels[:, 4] / 2
    return [
        CV2FormatLabel(min_x, min_y, max_x, max_y, label)
        for min_x, min_y, max_x, max_y, label in zip(min_xs, min_ys, max_xs, max_ys, labels)
    ]


def read_cv2_format_labels(labels_path: Path) -> list[CV2FormatLabel]:
    yolo_format_labels = np.loadtxt(labels_path)
    if len(yolo_format_labels.shape) == 1:
        yolo_format_labels = yolo_format_labels[None, :]
    return labels_yolo_to_cv2_format(yolo_format_labels)


def _overlay_label_on_image(
    img: np.ndarray, label: CV2FormatLabel, color: Color = BLUE
) -> np.ndarray:
    height, width, _ = img.shape
    min_x = int(label.normalized_min_x * width)
    min_y = int(label.normalized_min_y * height)
    max_x = int(label.normalized_max_x * width)
    max_y = int(label.normalized_max_y * height)
    if label.label_code != 0:
        scale = 0.7 * (height + width) / 2 / 512
        img = cv2.putText(
            img,
            LABEL_CODE_TO_NAME[label.label_code],
            (min_x, min_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            WHITE,
            2,
        )
    img = cv2.rectangle(img, (min_x, min_y), (max_x, max_y), color, 3)
    return img


def overlay_labels_on_image(img: np.ndarray, labels: list[CV2FormatLabel]) -> np.ndarray:
    for label in labels:
        img = _overlay_label_on_image(img, label)
    return img


def overlay_detections_on_image(
    img: np.ndarray, detections: list[CV2FormatLabel], correct: list[bool]
) -> np.ndarray:
    for i, detection in enumerate(detections):
        color = GREEN if correct[i] else RED
        img = _overlay_label_on_image(img, detection, color)
    return img
