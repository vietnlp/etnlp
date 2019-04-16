# from etnlp_api import embedding_config
from etnlp_api import embedding_extractor


INPUT_FILES = "/Users/sonvx/Documents/Pretrained_Models/W2V_C2V.vec"
              #   \
              # ";/Users/sonvx/Documents/Pretrained_Models/Bert_Large.vec;" \
              # "/Users/sonvx/Documents/Pretrained_Models/MULTI.vec"
# INPUT_FILES = "../data/embedding_dicts/W2V_100.vec;" \
#               "../data/embedding_dicts/ELMO_100.vec;" \
#               "../data/embedding_dicts/FastText_100.vec"
C2V = "../data/embedding_dicts/C2V_100.vec"
OUTPUT = "../data/embedding_dicts/MULTI_W_E_F.vec"
VOCAB_FILE = "../data/vocab.txt"
# OUTPUT_FORMAT=".txt;.npz;.gz"
OUTPUT_FORMAT = ".txt"
# embedding_config
embedding_extractor.extract_embedding_for_vocab_file(INPUT_FILES, VOCAB_FILE,
                                                     C2V, OUTPUT, OUTPUT_FORMAT)
print("DONE")

