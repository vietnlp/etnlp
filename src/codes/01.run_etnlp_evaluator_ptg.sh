#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$PWD"
INPUT_FILES="../data/embedding_dicts/Portuguese/fasttext_cbow_s300_analogy_POSTAG_vocab.vec;../data/embedding_dicts/Portuguese/w2v_cbow_s300_analogy_POSTAG_vocab.vec"
ANALOGY_FILE="../data/embedding_analogies/portuguese/LX-4WAnalogies-ETNLP.txt"
OUT_FILE="../data/embedding_analogies/portuguese/evaluator_results.txt"
python ./etnlp_api.py  -input $INPUT_FILES -output $OUT_FILE -analoglist $ANALOGY_FILE -args eval
