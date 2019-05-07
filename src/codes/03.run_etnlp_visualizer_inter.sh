#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$PWD"
INPUT_FILES="../data/embedding_dicts/ELMO_23.vec;../data/embedding_dicts/FastText_23.vec;../data/embedding_dicts/W2V_C2V_23.vec;../data/embedding_dicts/MULTI_23.vec"
python ./etnlp_api.py  -input $INPUT_FILES -args visualizer -port 8889
