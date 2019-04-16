from etnlp_api import embedding_config
from etnlp_api import embedding_extractor


emb1 = "<point_to_your_downloaded_file>/W2V_C2V.vec"
emb2 = "<point_to_your_downloaded_file>/ELMO.vec"
emb3 = "<point_to_your_downloaded_file>/MULTI.vec"
emb4 = "<point_to_your_downloaded_file>/FastText.vec"
C2V = "../data/embedding_dicts/C2V.vec"
out1 = "../data/embedding_dicts/W2V_C2V_23.vec"
out2 = "../data/embedding_dicts/ELMO_23.vec"
out3 = "../data/embedding_dicts/MULTI_23.vec"
out4 = "../data/embedding_dicts/FastText_23.vec"

VOCAB_FILE = "../data/vocab.txt"
# OUTPUT_FORMAT=".txt;.npz;.gz"
OUTPUT_FORMAT = ".txt"
# embedding_config
embedding_config.do_normalize_emb = True

emb_files = [emb1, emb2, emb3, emb4]
out_files = [out1, out2, out3, out4]

for emb_file, out_file in zip(emb_files, out_files):
    embedding_extractor.extract_embedding_for_vocab_file(emb_file, VOCAB_FILE,
                                                     C2V, out_file, OUTPUT_FORMAT)
print("DONE")

