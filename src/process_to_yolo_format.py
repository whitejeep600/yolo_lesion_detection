from pathlib import Path

import pandas as pd


EVAL = "eval"
TRAIN = "train"
TEST = "test"
splits = (TRAIN, EVAL, TEST)

IMAGES = "images"
LABELS = "labels"


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


def main(
        raw_images_dir: Path,
        metadata_path: Path,
        target_path: Path
) -> None:
    target_path.mkdir(parents=True, exist_ok=True)
    split_dirs = {
        split: target_path / split for split in splits
    }
    for split_dir in split_dirs.values():
        split_dir.mkdir(parents=True, exist_ok=True)
    image_dirs = {
        split: split_dirs[split] / IMAGES for split in splits
    }
    label_dirs = {
        split: split_dirs[split] / LABELS for split in splits
    }
    for dirs in image_dirs, label_dirs:
        for split_dir in dirs.values():
            split_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(metadata_path)
    img_id_to_labels, img_id_to_split = get_labels_and_splits_from_meta(meta)

    for batch_dir in raw_images_dir.iterdir():
        batch_subdir = batch_dir / "Images_png"
        for series_subdir in batch_subdir.iterdir():
            patient, study, series = str(series_subdir.name).split("_")
            for slice_file in series_subdir.iterdir():
                slice_no = slice_file.stem
                img_id = f"{patient}_{study}_{series}_{slice_no}"
                if img_id not in img_id_to_split.keys():
                    continue
                print(slice_file)
                img_labels = img_id_to_labels[img_id]
                img_split = img_id_to_split[img_id]
                print(img_id)
                # copy the image to the target location
                # write labels
                #   detype labels if they're in train or eval

    # save auxiliary information to data/yolo_format/{train,eval}/split_data.csv
    pass


if __name__ == '__main__':
    raw_images_dir = Path("data/raw/images")
    metadata_path = Path("data/raw/DL_info.csv")
    target_path = Path("data/yolo_format")
    main(
        raw_images_dir,
        metadata_path,
        target_path
    )
