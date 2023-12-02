import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.constants import TRAIN, EVAL, TEST, LABELS, IMAGES, SPLITS

# In the original data, the numbers 1, 2, 3 correspond to train, eval and test splits,
# respectively. However, only eval and test are annotated. I don't know why. No, really,
# I am really curious why a design choice like that was made. One obvious explanation is
# to save money on annotations. But then why bother labeling anything at all if it's
# impossible to train a lesion classifier based on the data anyway?
METADATA_SPLIT_NO_TO_SPLIT = {
    1: TRAIN,
    2: EVAL,
    3: TEST
}


# Lesion type data is only available for the eval and test set. We cannot train a
# lesion classifier, so we will only train an untyped lesion detector. For consistency
# between the train and eval splits, we remove labels from the eval split as well.
# The labels will still be used on the test set, but only to compare which lesion
# types are easier/harder to detect.
class YoloFormatLabel:
    def __init__(
            self,
            normalized_center_x: float,
            normalized_center_y: float,
            normalized_width: float,
            normalized_height: float,
            label: int
    ):
        self.center_x = normalized_center_x
        self.center_y = normalized_center_y
        self.width = normalized_width
        self.height = normalized_height
        self.label = label

    def to_text(self, preserve_type: bool):
        label = self.label if preserve_type else 0
        return f"{label} {self.center_x} {self.center_y} {self.width} {self.height}"


def metadata_to_label(meta: pd.Series) -> YoloFormatLabel:
    img_width, img_height = [int(x) for x in meta["Image_size"].split(",")]
    x_min, y_min, x_max, y_max = [float(x) for x in meta["Bounding_boxes"].split(",")]
    x_min /= img_width
    x_max /= img_width
    y_min /= img_height
    y_max /= img_height
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return YoloFormatLabel(
        x_center, y_center, width, height, meta["Coarse_lesion_type"]
    )


def get_labels_and_splits_from_meta(
        meta: pd.DataFrame
) -> tuple[dict[str, list[YoloFormatLabel]], dict[str, str]]:
    img_ids = [filename[:-4] for filename in meta["File_name"]]
    img_id_to_labels = {
        img_id: []
        for img_id in img_ids
    }
    img_id_to_split = {}
    for row in meta.iterrows():
        row_data = row[1]
        img_id = row_data["File_name"][:-4]
        img_id_to_labels[img_id].append(metadata_to_label(row_data))
        split = METADATA_SPLIT_NO_TO_SPLIT[row_data["Train_Val_Test"]]
        img_id_to_split[img_id] = split
    return img_id_to_labels, img_id_to_split


def write_labels_in_yolo_format(
        labels: list[YoloFormatLabel],
        target_path: Path,
        preserve_type: bool
) -> None:
    label_text_representations = [
        label.to_text(preserve_type) for label in labels
    ]
    all_representations = "\n".join(label_text_representations)
    with open(target_path, "w") as target_file:
        target_file.write(all_representations)
    pass


def main(
        raw_images_dir: Path,
        metadata_path: Path,
        target_path: Path
) -> None:
    target_path.mkdir(parents=True, exist_ok=True)
    split_dirs = {
        split: target_path / split for split in SPLITS
    }
    for split_dir in split_dirs.values():
        split_dir.mkdir(parents=True, exist_ok=True)
    image_dirs = {
        split: split_dirs[split] / IMAGES for split in SPLITS
    }
    label_dirs = {
        split: split_dirs[split] / LABELS for split in SPLITS
    }
    for dirs in image_dirs, label_dirs:
        for split_dir in dirs.values():
            split_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(metadata_path)
    img_id_to_labels, img_id_to_split = get_labels_and_splits_from_meta(meta)

    n_batches = len(list(raw_images_dir.iterdir()))
    for batch_dir in tqdm(raw_images_dir.iterdir(), total=n_batches):
        batch_subdir = batch_dir / "Images_png"
        for series_subdir in batch_subdir.iterdir():
            patient, study, series = str(series_subdir.name).split("_")
            for slice_file in series_subdir.iterdir():
                slice_no = slice_file.stem
                img_id = f"{patient}_{study}_{series}_{slice_no}"
                if img_id not in img_id_to_split.keys():
                    continue
                img_labels = img_id_to_labels[img_id]
                img_split = img_id_to_split[img_id]

                target_image_path = image_dirs[img_split] / f"{img_id}.png"
                os.system(f"cp {slice_file} {target_image_path}")

                untype_or_not = img_split == TEST
                target_labels_path = label_dirs[img_split] / f"{img_id}.txt"
                write_labels_in_yolo_format(
                    img_labels,
                    target_labels_path,
                    untype_or_not
                )


if __name__ == '__main__':
    raw_images_dir = Path("data/raw/images")
    metadata_path = Path("data/raw/DL_info.csv")
    target_path = Path("data/yolo_format")
    main(
        raw_images_dir,
        metadata_path,
        target_path
    )
