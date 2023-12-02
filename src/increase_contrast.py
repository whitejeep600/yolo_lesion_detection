from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.constants import SPLITS, IMAGES


def increase_contrast(
        img: np.ndarray
) -> np.ndarray:
    img = img.astype(float)
    # Simply normalize the range of pixel image values to [0, 255].
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(int)


def main(dataset_path: Path):
    split_paths = [dataset_path / split / IMAGES for split in SPLITS]
    for split_path in split_paths:
        print(f"Increasing contrast for split {split_path.parent.name}")
        n_total = len(list(split_path.iterdir()))
        for img_path in tqdm(split_path.iterdir(), total=n_total):
            low_contrast_img = cv2.imread(str(img_path))
            high_contrast_img = increase_contrast(low_contrast_img)
            cv2.imwrite(str(img_path), high_contrast_img)


if __name__ == '__main__':
    dataset_path = Path("data/yolo_format")
    main(dataset_path)
