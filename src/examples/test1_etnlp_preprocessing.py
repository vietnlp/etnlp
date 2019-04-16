from etnlp_api import embedding_preprocessing as emb_prep
from etnlp_api import embedding_config

INPUT_FILES="../data/glove2vec_dicts/glove1.vec;../data/glove2vec_dicts/glove2.vec"
OUTPUT_FILES="../data/glove2vec_dicts/glove1_w2v.vec;../data/glove2vec_dicts/glove2_w2v.vec"
# do_normalize: use this flag to normalize in case of multiple embeddings.
embedding_config.do_normalize_emb = False
# to mark input embeddings are not in word2vec format.
embedding_config.is_word2vec_format = False
emb_prep.load_and_save_2_word2vec_models(INPUT_FILES, OUTPUT_FILES, embedding_config)

print("Done with exporting")