from etnlp_api import embedding_evaluator
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

INPUT_FILES = "../data/embedding_dicts/ELMO_100.vec;FastText_100.vec"
ANALOGY_FILE = "../data/embedding_analogies/vi/solveable_analogies_vi.txt"
OUT_FILE = "../data/embedding_analogies/vi/Multi_evaluator_results.txt"
embedding_evaluator.evaluator_api(INPUT_FILES, ANALOGY_FILE, OUT_FILE)
print("DONE")

