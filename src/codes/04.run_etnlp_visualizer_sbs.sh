#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$PWD"
INPUT_FILES="/Users/sonvx/Documents/Pretrained_Models/W2V.vec;/Users/sonvx/Documents/Pretrained_Models/W2V_C2V.vec;/Users/sonvx/Documents/Pretrained_Models/FastText.vec;/Users/sonvx/Documents/Pretrained_Models/ELMO.vec;/Users/sonvx/Documents/Pretrained_Models/Bert_Base.vec;/Users/sonvx/Documents/Pretrained_Models/Bert_Large.vec;/Users/sonvx/Documents/Pretrained_Models/MULTI.vec"
# python ./visualizer/Main.py  -input $INPUT_FILES -args visualizer
python ./visualizer/Main.py $INPUT_FILES
