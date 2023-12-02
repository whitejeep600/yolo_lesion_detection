from pathlib import Path

import cv2
import numpy as np

from src.constants import SPLITS, IMAGES, LABELS


LABEL_INT_TO_DESCRIPTION = {
    1: "bone",
    2: "abdomen",
    3: "mediastinum",
    4: "liver",
    5: "lung",
    6: "kidney",
    7: "soft",
    8: "pelvis"
}


class CV2FormatLabel:
    def __init__(
            self,
            normalized_min_x: float,
            normalized_min_y: float,
            normalized_max_x: float,
            normalized_max_y: float,
            label: int
    ):
        self.normalized_min_x = normalized_min_x
        self.normalized_min_y = normalized_min_y
        self.normalized_max_x = normalized_max_x
        self.normalized_max_y = normalized_max_y
        self.label = label


def _labels_yolo_to_cv2_format(
        yolo_format_labels: np.ndarray
) -> list[CV2FormatLabel]:
    labels = yolo_format_labels[:, 0]
    min_xs = yolo_format_labels[:, 1] - yolo_format_labels[:, 3] / 2
    min_ys = yolo_format_labels[:, 2] - yolo_format_labels[:, 4] / 2
    max_xs = yolo_format_labels[:, 1] + yolo_format_labels[:, 3] / 2
    max_ys = yolo_format_labels[:, 2] + yolo_format_labels[:, 4] / 2
    return [
        CV2FormatLabel(min_x, min_y, max_x, max_y, label)
        for min_x, min_y, max_x, max_y, label in zip(min_xs, min_ys, max_xs, max_ys, labels)
    ]


def _read_cv2_format_labels(
        labels_path: Path
) -> list[CV2FormatLabel]:
    yolo_format_labels = np.loadtxt(labels_path)
    if len(yolo_format_labels.shape) == 1:
        yolo_format_labels = yolo_format_labels[None, :]
    return _labels_yolo_to_cv2_format(yolo_format_labels)


def _overlay_label_on_image(
        img: np.ndarray,
        label: CV2FormatLabel
) -> np.ndarray:
    height, width, _ = img.shape
    min_x = int(label.normalized_min_x * width)
    min_y = int(label.normalized_min_y * height)
    max_x = int(label.normalized_max_x * width)
    max_y = int(label.normalized_max_y * height)
    img = cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
    return img


def _overlay_labels_on_image(
        img: np.ndarray,
        labels: list[CV2FormatLabel]
) -> np.ndarray:
    for label in labels:
        img = _overlay_label_on_image(img, label)
    return img


def _overlay_labels_on_split(
        split_path: Path,
        target_dir: Path
):
    target_dir.mkdir(exist_ok=True, parents=True)
    images_dir = split_path / IMAGES
    labels_dir = split_path / LABELS
    for img_path in images_dir.iterdir():
        img_id = img_path.stem
        labels_path = labels_dir / f"{img_id}.txt"
        labels = _read_cv2_format_labels(labels_path)
        img = cv2.imread(str(img_path))
        img = _overlay_labels_on_image(img, labels)
        target_plot_path = target_dir / img_path.name
        cv2.imwrite(str(target_plot_path), img)


def main(dataset_path: Path, target_path: Path):
    target_path.mkdir(exist_ok=True, parents=True)
    for split in SPLITS:
        _overlay_labels_on_split(dataset_path / split, target_path / split)


if __name__ == '__main__':
    dataset_path = Path("data/yolo_format")
    target_path = Path("plots")
    main(dataset_path, target_path)


