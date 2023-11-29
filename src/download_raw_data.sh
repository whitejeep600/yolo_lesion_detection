mkdir -p data/raw
python -m src/batch_download_zips

for f in data/raw/* ; do unzip "$f" -d "${f%.*}" ; done