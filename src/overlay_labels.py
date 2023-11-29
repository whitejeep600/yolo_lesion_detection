from pathlib import Path

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


def overlay_labels_on_split(
        split_path: Path,
        target_path: Path
):
    target_path.mkdir(exist_ok=True, parents=True)
    images_dir = split_path / IMAGES
    labels_dir = split_path / LABELS
    for img_path in images_dir.iterdir():
        img_id = img_path.stem
        labels_path = labels_dir / f"{img_id}.txt"
        labels = np.loadtxt(labels_path)
        print(labels)


def main(dataset_path: Path, target_path: Path):
    target_path.mkdir(exist_ok=True, parents=True)
    for split in SPLITS:
        overlay_labels_on_split(dataset_path / split, target_path / split)


if __name__ == '__main__':
    dataset_path = Path("data/yolo_format")
    target_path = Path("plots")
    main(dataset_path, target_path)


