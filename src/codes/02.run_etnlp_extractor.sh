#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$PWD"
INPUT_FILES="../data/embedding_dicts/ELMO_23.vec;../data/embedding_dicts/FastText_23.vec;../data/embedding_dicts/W2V_C2V_23.vec;../data/embedding_dicts/MULTI_23.vec"
C2V="../data/embedding_dicts/C2V.vec"
OUTPUT="../data/embedding_dicts/MULTI_W_F_B_E.vec"
VOCAB_FILE="../data/vocab.txt"
python ./etnlp_api.py  -input $INPUT_FILES -vocab $VOCAB_FILE -input_c2v $C2V -args "extract" -output $OUTPUT
