# Convert to a standard word2vec format

import gensim
from utils import embedding_io
import sys
from threading import Thread
from embeddings.embedding_configs import EmbeddingConfigs


def convert_to_w2v(vocab_file, embedding_file, out_file):
    """
    Export from a word2vec file by filtering out vocabs based on the input vocab file.
    :param vocab_file:
    :param embedding_file:
    :param out_file:
    :return: word2vec file
    """
    std_vocab = []
    with open(vocab_file) as f:
        for word in f:
            std_vocab.append(word)

    print ("Loaded NER vocab_size = %s" % (len(std_vocab)))
    is_binary = False
    if embedding_file.endswith(".bin"):
        is_binary = True

    print("Loading w2v model ...")

    emb_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file,
                                                                binary=is_binary,
                                                                unicode_errors='ignore')

    print("LOADED model: vocab_size = %s" % (len(emb_model.wv.vocab)))
    f_writer = open(out_file, "w")
    for word in std_vocab:
        word = word.rstrip()
        line = None
        if word in emb_model:
            vector = " ".join(str(item) for item in emb_model[word])
            # word = word.lower()
            line = "%s %s" % (word, vector)
        else:
            word = word.lower()
            if word in emb_model:
                vector = " ".join(str(item) for item in emb_model[word])
                line = "%s %s" % (word, vector)
                # print("LINE: ", line)
        if line:
            f_writer.write(line + "\n")
    f_writer.close()


def test():
    vocab_file = "../data/vnner_BiLSTM_CRF/vocab.words.txt"
    embedding_file = "../data/embedding_dicts/elmo_embeddings_large.txt"
    out_file = "../data/embedding_dicts/elmo_1024dims_wiki_normalcase2lowercase_NER.vec"
    convert_to_w2v(vocab_file, embedding_file, out_file)
    print("Out file: ", out_file)
    print("DONE")


def load_and_save_2_word2vec_model(input_model_path, output_model_path, embedding_config):
    """
    Process one embedding model
    :param input_model_path:
    :param output_model_path:
    :return:
    """
    model_in = embedding_io.load_word_embedding(input_model_path, embedding_config)
    embedding_io.save_model_to_file(model_in, output_model_path)
    print("Write model back to ", output_model_path)


def load_and_save_2_word2vec_models(input_embedding_files_str, output_embedding_files_str, embedding_config):
    """
    Multi-threaded processing to export to word2vec format
    :param input_embedding_files_str:
    :param output_embedding_files_str:
    :return:
    """
    if input_embedding_files_str.__contains__(";"):
        input_model_files = input_embedding_files_str.split(";")
    else:
        input_model_files = [input_embedding_files_str]

    if output_embedding_files_str.__contains__(";"):
        output_model_files = output_embedding_files_str.split(";")
    else:
        output_model_files = [output_embedding_files_str]

    # Double check input files and output files.
    assert (len(output_model_files) == len(input_model_files)), \
        "Number of input files and output files must be equal. Exiting ..."

    # create a list of threads
    threads = []

    for model_in, model_out in zip(input_model_files, output_model_files):
        # We start one thread per file.
        process = Thread(target=load_and_save_2_word2vec_model, args=[model_in, model_out, embedding_config])
        process.start()
        threads.append(process)
        # load_and_save_2_word2vec_model(model_in, model_out)

    # This to ensure each thread has finished processing the input file.
    for process in threads:
        process.join()


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Missing input arguments. Input format: ./*.py <emb_file1;emb_file2;...>. Exiting ...")
        exit(0)

    embedding_config = EmbeddingConfigs()
    # We don't need to be word2vec format for pre-processing here but it still shows warning
    # if input files aren't in w2v format.
    embedding_config.is_word2vec_format = True
    embedding_config.do_normalize_emb = False # If you don't want to normalize the embedding vectors.

    if sys.argv[1].__contains__(";"):
        in_model_files = sys.argv[1].split(";")
    else:
        in_model_files = [sys.argv[1]]

    out_model_files = [input_model_path + ".extracted.vec" for input_model_path in in_model_files]

    load_and_save_2_word2vec_models(in_model_files, out_model_files)
