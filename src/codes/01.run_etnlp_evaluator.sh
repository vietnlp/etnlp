#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$PWD"
INPUT_FILES="../data/embedding_dicts/ELMO_23.vec;../data/embedding_dicts/FastText_23.vec;../data/embedding_dicts/W2V_C2V_23.vec;../data/embedding_dicts/MULTI_23.vec"
ANALOGY_FILE="../data/embedding_analogies/vi/solveable_analogies_vi.txt"
OUT_FILE="../data/embedding_analogies/vi/Multi_evaluator_results.txt"
python ./etnlp_api.py  -input $INPUT_FILES -output $OUT_FILE -analoglist $ANALOGY_FILE -args eval