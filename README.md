# yolo_lesion_detection

This project was developed as an assignment for the Medical Image Processing course at the
National Taiwan University. Below are instructions on how to reproduce the results. For a
description of the project's goals and findings, refer to `report.md`.

Run

`$ src/download_raw_data.sh`

to get the unprocessed DeepLesion dataset. In addition, after agreeing to DeepLesion's usage
terms, manually download the DL_info.csv file into the data/raw directory. Afterwards, run

`$ python -m src.process_to_yolo_format`

to set up yolo-format train, eval and test splits, with labels gathered into one file for each
image, and only images with at least one lesion. To verify the correctness of the labels,
they can be overlaid with the `stc/overlay_labels.py` script.

After this step, it is necessary to increase  the contrast of the images to make them look more
natural (to the human eye and to YOLO), since the original images are saved in a peculiar
format with a very narrow range of pixel intensities. The original data provides a FAQ with
instructions on how to increase the intensity; however, these instructions don't make sense,
(or at least I can't make sense of them), so I wrote a simplistic custom algorithm.

Before training, clone into the [yolov5 repository](https://github.com/ultralytics/yolov5) so that
it is a subdirectory of `yolo_lesion_detection`. After this step, training can be launched
with

`$ bash src/train_yolo.sh`

If the resulting weights are saved as `weights/yolov5.pt` (or otherwise the relevant path
is adjusted in `src/evaluate.py`), it is possible to run the evaluation on the test set,
which will produce the plots of all the images from the test set (with overlaid labels and
the model's detections), as well as save some evaluation metrics.