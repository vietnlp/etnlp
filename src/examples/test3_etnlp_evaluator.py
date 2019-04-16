from etnlp_api import embedding_evaluator
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

INPUT_FILES = "../data/embedding_dicts/ELMO_23.vec;../data/embedding_dicts/FastText_23.vec;" \
              "../data/embedding_dicts/W2V_C2V_23.vec;../data/embedding_dicts/MULTI_23.vec"
ANALOGY_FILE = "../data/embedding_analogies/vi/solveable_analogies_vi.txt"
OUT_FILE = "../data/embedding_analogies/vi/Multi_evaluator_results.txt"
embedding_evaluator.evaluator_api(INPUT_FILES, ANALOGY_FILE, OUT_FILE)
print("DONE")