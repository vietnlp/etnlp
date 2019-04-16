#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$PWD"
# python ./api/embedding_preprocessing.py
# python ./api/embedding_evaluator.py  -m /Users/sonvx/Documents/Pretrained_Models/elmo_embeddings.txt -v True -d 1024
# python ./api/embedding_evaluator.py  -m /Users/sonvx/Documents/Pretrained_Models/fastText_wiki_lowercase_300_NER.vec -v True -d 300
# python ./api/embedding_evaluator.py  -m /Users/sonvx/Documents/Pretrained_Models/w2v_wiki_lowercase_300_NER.vec -v True -d 300 -lowercase True
# python ./api/embedding_evaluator.py  -m /Users/sonvx/Documents/Pretrained_Models/w2v_c2v_wiki_lowercase_300_NER.vec -v True -d 300 -lowercase True
# python ./api/embedding_evaluator.py  -m /Users/sonvx/Documents/Pretrained_Models/bert_base_wiki_lowercase_768_NER_wikicontexts.vec -v True -d 768 -lowercase True
# python ./api/embedding_evaluator.py  -m /Users/sonvx/Documents/Pretrained_Models/bert_large_wiki_lowercase_1024_NER_wikicontexts.vec -v True -d 1024 -lowercase True
# Elmo normal2lowercase: check normal then check lower; _2 means remove redundancy
INPUT_FILES="/Users/sonvx/Documents/Pretrained_Models/fastText_wiki_lowercase_300_NER.vec;/Users/sonvx/Documents/Pretrained_Models/Dec07_multi_fastText_W2V_with_c2v4oov_lowercase_Wiki_elmo_300_300_1024_rm0.vec"
ANALOGY_FILE="../data/embedding_analogies/vi/solveable_analogies_vi.txt"
OUT_FILE="../data/embedding_analogies/vi/elmo_results_out_dict.txt"
# python ./api/embedding_evaluator.py  -input $INPUT_FILES -output $OUT_FILE -analoglist $ANALOGY_FILE
python ./etnlp_api.py  -input $INPUT_FILES -output $OUT_FILE -analoglist $ANALOGY_FILE
# OUT_FILE="../data/embedding_analogies/vi/fastText_results_out_2.txt"
# python ./api/embedding_evaluator.py  -m /Users/sonvx/Documents/Pretrained_Models/fastText_wiki_lowercase_300_NER.vec -v True -d 300 -lowercase False -results_file $OUT_FILE
# BertBase
# OUT_FILE="../data/embedding_analogies/vi/bert_Base_results_out_unique_all.txt"
# python ./api/embedding_evaluator.py  -m /Users/sonvx/Documents/Pretrained_Models/bert_base_wiki_lowercase_768_NER_wikicontexts.vec -v True -d 768 -lowercase False -results_file $OUT_FILE -remove_redundancy True
# BertLarge
# OUT_FILE="../data/embedding_analogies/vi/bert_Large_results_out_dict.txt"
# EMB_FILE="/Users/sonvx/Documents/Pretrained_Models/bert_large_wiki_lowercase_1024_NER_wikicontexts.vec"
# python ./api/embedding_evaluator.py  -m $EMB_FILE -v True -d 1024 -lowercase False -results_file $OUT_FILE -remove_redundancy False
## W2V
# OUT_FILE="../data/embedding_analogies/vi/W2V_results_out_unique_rm_redun.txt"
# EMB_FILE="/Users/sonvx/Documents/Pretrained_Models/w2v_wiki_lowercase_300_NER.vec"
# python ./api/embedding_evaluator.py  -m $EMB_FILE -v True -d 300 -lowercase False -results_file $OUT_FILE -remove_redundancy True
## FastText
# OUT_FILE="../data/embedding_analogies/vi/FastText_results_out_dict.txt"
# EMB_FILE="/Users/sonvx/Documents/Pretrained_Models/fastText_wiki_lowercase_300_NER.vec"
# python ./api/embedding_evaluator.py  -m $EMB_FILE -v True -d 300 -lowercase False -results_file $OUT_FILE -remove_redundancy False
## W2V_c2v
# OUT_FILE="../data/embedding_analogies/vi/W2V_c2v_results_out_dict.txt"
# EMB_FILE="/Users/sonvx/Documents/Pretrained_Models/w2v_c2v_wiki_lowercase_300_NER.vec"
# python ./api/embedding_evaluator.py  -m $EMB_FILE -v True -d 300 -lowercase False -results_file $OUT_FILE -remove_redundancy False
## MULTI
# OUT_FILE="../data/embedding_analogies/vi/multi_W2V_c2v_fasttext_elmo_results_out_dict.txt"
# EMB_FILE="../data/embedding_dicts/Dec07_multi_fastText_W2V_with_c2v4oov_lowercase_Wiki_elmo_300_300_1024_rm0.vec"
# python ./api/embedding_evaluator.py  -m $EMB_FILE -v True -d 1624 -lowercase False -results_file $OUT_FILE -remove_redundancy False

