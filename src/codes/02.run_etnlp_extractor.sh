#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$PWD"
INPUT_FILES="/Users/sonvx/Documents/Pretrained_Models/W2V.vec;/Users/sonvx/Documents/Pretrained_Models/ELMO.vec;/Users/sonvx/Documents/Pretrained_Models/Bert_Base.vec;/Users/sonvx/Documents/Pretrained_Models/FastText.vec"
C2V="/Users/sonvx/Documents/Pretrained_Models/C2V.vec"
OUTPUT="../data/embedding_dicts/MULTI_W_F_B_E.vec"
VOCAB_FILE="../data/vocab.txt"
python ./etnlp_api.py  -input $INPUT_FILES -vocab $VOCAB_FILE -input_c2v $C2V -args "extract" -output $OUTPUT
