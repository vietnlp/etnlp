# from etnlp_api import embedding_config
from etnlp_api import embedding_visualizer

INPUT_FILES = "../data/embedding_dicts/ELMO_12.vec;../data/embedding_dicts/FastText_12.vec;" \
              "../data/embedding_dicts/W2V_C2V_12.vec;../data/embedding_dicts/MULTI_12.vec"
embedding_visualizer.visualize_multiple_embeddings(INPUT_FILES)

print("DONE")