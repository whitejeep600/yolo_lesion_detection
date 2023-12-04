from pathlib import Path

import cv2

from src.constants import IMAGES, LABELS, SPLITS
from src.label_processing import overlay_labels_on_image, read_cv2_format_labels


def _overlay_labels_on_split(split_path: Path, target_dir: Path):
    target_dir.mkdir(exist_ok=True, parents=True)
    images_dir = split_path / IMAGES
    labels_dir = split_path / LABELS
    for img_path in images_dir.iterdir():
        img_id = img_path.stem
        labels_path = labels_dir / f"{img_id}.txt"
        labels = read_cv2_format_labels(labels_path)
        img = cv2.imread(str(img_path))
        img = overlay_labels_on_image(img, labels)
        target_plot_path = target_dir / img_path.name
        cv2.imwrite(str(target_plot_path), img)


def main(dataset_path: Path, target_path: Path):
    target_path.mkdir(exist_ok=True, parents=True)
    for split in SPLITS:
        _overlay_labels_on_split(dataset_path / split, target_path / split)


if __name__ == "__main__":
    dataset_path = Path("data/yolo_format")
    target_path = Path("plots")
    main(dataset_path, target_path)
