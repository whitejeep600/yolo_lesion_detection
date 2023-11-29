python -m src/batch_download_zips

for f in data/raw/images/* ; do unzip "$f" -d "${f%.*}" ; done
