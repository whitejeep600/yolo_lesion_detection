from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from src.constants import IMAGES, LABEL_CODE_TO_NAME, LABELS
from src.label_processing import (
    CV2FormatLabel,
    labels_overlap,
    labels_yolo_to_cv2_format,
    overlay_detections_on_image,
    overlay_labels_on_image,
    read_cv2_format_labels,
)


def _get_cv2_format_detections(img_path: Path, model: torch.nn.Module) -> list[CV2FormatLabel]:
    raw_detections = model(img_path)
    yolo_format_array = raw_detections.xywhn[0].detach().numpy()
    yolo_format_array = yolo_format_array[:, [5, 0, 1, 2, 3]]
    detections = labels_yolo_to_cv2_format(yolo_format_array)
    return detections


def main(
    dataset_path: Path,
    weights_path: Path,
    target_plots_dir: Path,
    target_metrics_path: Path,
):
    target_plots_dir.mkdir(exist_ok=True, parents=True)
    model = torch.hub.load(
        "yolov5/",
        "custom",
        path=weights_path,
        source="local",
    )
    images_dir = dataset_path / IMAGES
    labels_dir = dataset_path / LABELS
    n_total = len(list(images_dir.iterdir()))
    n_false_positives = 0
    n_all_detections = 0
    label_code_to_n_detected = {label_code: 0 for label_code in LABEL_CODE_TO_NAME}
    label_code_to_n_total = {label_code: 0 for label_code in LABEL_CODE_TO_NAME}
    for img_path in tqdm(images_dir.iterdir(), total=n_total, desc="Running the model on images"):
        img_id = img_path.stem
        labels_path = labels_dir / f"{img_id}.txt"
        labels = read_cv2_format_labels(labels_path)
        detections = _get_cv2_format_detections(img_path, model)

        # Obviously this should be vectorized but we mostly have 1 label per
        # image so I didn't bother. To this crime I confess
        overlaps = [
            [labels_overlap(label, detection) for label in labels] for detection in detections
        ]
        detection_correct = [
            any(overlaps[i][j] for j in range(len(labels))) for i in range(len(detections))
        ]
        detected_labels = [
            any(overlaps[i][j] for i in range(len(detections))) for j in range(len(labels))
        ]
        n_false_positives += len(detection_correct) - sum(detection_correct)
        n_all_detections += len(detections)
        for i in range(len(labels)):
            label_code_to_n_total[labels[i].label_code] += 1
            if detected_labels[i]:
                label_code_to_n_detected[labels[i].label_code] += 1
        img = cv2.imread(str(img_path))
        img = overlay_labels_on_image(img, labels)
        img = overlay_detections_on_image(img, detections, detection_correct)
        target_path = target_plots_dir / img_path.name
        cv2.imwrite(str(target_path), img)

    false_positive_rate = n_false_positives / n_all_detections
    per_class_recall = {
        LABEL_CODE_TO_NAME[label_code]: label_code_to_n_detected[label_code]
        / label_code_to_n_total[label_code]
        if label_code_to_n_total[label_code] != 0
        else None
        for label_code in LABEL_CODE_TO_NAME
    }
    general_recall = sum(label_code_to_n_detected.values()) / sum(label_code_to_n_total.values())

    with open(target_metrics_path, "w") as result_file:
        result_file.write(
            f"False positive rate {false_positive_rate}, general recall {general_recall},\n"
            f"recall per class: {per_class_recall}"
        )


if __name__ == "__main__":
    dataset_path = Path("data/yolo_format/test")
    weights_path = Path("weights/yolov5.pt")
    target_plots_dir = Path("evaluation/plots")
    target_metrics_path = Path("evaluation/metrics.txt")
    main(dataset_path, weights_path, target_plots_dir, target_metrics_path)
