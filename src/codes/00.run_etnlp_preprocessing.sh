#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$PWD"
INPUT_FILES="../data/glove2vec_dicts/glove1.vec;../data/glove2vec_dicts/glove2.vec"
OUTPUT_FILES="../data/glove2vec_dicts/glove1_w2v.vec;../data/glove2vec_dicts/glove2_w2v.vec"
# do_normalize: use this flag to normalize in case of multiple embeddings.
python ./etnlp_api.py  -input $INPUT_FILES -output $OUTPUT_FILES -args "glove2w2v"
