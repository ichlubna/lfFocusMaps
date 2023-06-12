#!/bin/bash\

wget https://merlin.fit.vutbr.cz/LightField/datasets/lfDataset/data/blenderBmw.mkv
mkdir ./scene
ffmpeg -i blenderBmv.mkv scene/%04d.png
mkdir ./sorted
python renameLFImages.py ./scene ./sorted 0 0 15 15 0 1 27
python measure.py ./sorted/data/ ./sorted/reference 0.03_0.4 8 8 1.9157
