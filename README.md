# yolo_lesion_detection

todo

write evaluation script (use label overlaying script)
generate some metrics, plots
download a ~100-img sample of the test set locally, run
run on the real test split
write a report



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