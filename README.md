# yolo_lesion_detection

todo

increase contrast on pepper
write label overlaying script
write yolo training 
launch training on pepper
write evaluation script (use label overlaying script)

Run

`$ src/download_raw_data.sh`

to get the unprocessed DeepLesion dataset. In addition, after agreeing to DeepLesion's usage
terms, manually download the DL_info.csv file into the data/raw directory. Afterwards, run

`$ python -m src.process_to_yolo_format`

to set up yolo-format train, eval and test splits, with labels gathered into one file for each
image, and only images with at least one lesion.

After this step, it is necessary to increase  the contrast of the images to make them look more
natural (to the human eye and to YOLO),
since the original images are saved in a peculiar format with a very narrow range of pixel
intensities. The original data provides a FAQ with instructions on how to increase the intensity;
however, these instructions don't make sense, so I wrote a simplistic custom algorithm.