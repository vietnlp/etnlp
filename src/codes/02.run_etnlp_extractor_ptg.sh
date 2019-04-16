#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$PWD"
#INPUT_FILES="/Users/sonvx/MyDocuments/OProjects/VNNLP/VietNLP.com/ETNLP/ETNLP/data/embedding_dicts/Portuguese/w2v_cbow_s300.txt"
#INPUT_FILES="/Users/sonvx/MyDocuments/OProjects/VNNLP/VietNLP.com/ETNLP/ETNLP/data/embedding_dicts/Portuguese/fasttext_cbow_s300.txt"
# INPUT_FILES="/Users/sonvx/MyDocuments/OProjects/VNNLP/VietNLP.com/ETNLP/ETNLP/data/embedding_dicts/Portuguese/wang2vec_cbow_s300.txt"
INPUT_FILES="/Users/sonvx/MyDocuments/OProjects/VNNLP/VietNLP.com/ETNLP/ETNLP/data/embedding_dicts/Portuguese/glove_s300.txt"
# C2V="/Users/sonvx/Documents/Pretrained_Models/C2V.vec"
OUTPUT="../data/embedding_dicts/portuguese/glove_s300_analogy_POSTAG_vocab.vec"
# VOCAB_FILE="../data/embedding_analogies/portuguese/vocab.txt"
VOCAB_FILE="../data/embedding_analogies/portuguese/POST_TAG_vocabulary.txt"
python ./etnlp_api.py  -input $INPUT_FILES -vocab $VOCAB_FILE -args "extract" -output $OUTPUT
