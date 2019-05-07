#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$PWD"
INPUT_FILES="../data/embedding_dicts/ELMO_23.vec;../data/embedding_dicts/FastText_23.vec;../data/embedding_dicts/W2V_C2V_23.vec;../data/embedding_dicts/MULTI_23.vec"
# python ./visualizer/visualizer_sbs.py  -input $INPUT_FILES -args visualizer
python ./visualizer/visualizer_sbs.py $INPUT_FILES
