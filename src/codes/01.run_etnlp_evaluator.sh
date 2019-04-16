#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$PWD"
INPUT_FILES="/Users/sonvx/Documents/Pretrained_Models/W2V.vec;/Users/sonvx/Documents/Pretrained_Models/W2V_C2V.vec;/Users/sonvx/Documents/Pretrained_Models/FastText.vec;/Users/sonvx/Documents/Pretrained_Models/ELMO.vec;/Users/sonvx/Documents/Pretrained_Models/Bert_Base.vec;/Users/sonvx/Documents/Pretrained_Models/Bert_Large.vec;/Users/sonvx/Documents/Pretrained_Models/MULTI.vec;/Users/sonvx/MyDocuments/OProjects/VNNLP/VietNLP.com/ETNLP/ETNLP/data/embedding_dicts/MULTI_W_F_B_E.vec"
ANALOGY_FILE="../data/embedding_analogies/vi/solveable_analogies_vi.txt"
OUT_FILE="../data/embedding_analogies/vi/Multi_evaluator_results.txt"
python ./etnlp_api.py  -input $INPUT_FILES -output $OUT_FILE -analoglist $ANALOGY_FILE -args eval
