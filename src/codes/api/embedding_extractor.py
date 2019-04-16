from embeddings import embedding_utils
from pathlib import Path
import numpy as np
import os
import logging
import gzip
from embeddings.embedding_configs import EmbeddingConfigs


def get_multi_embedding_models(config: EmbeddingConfigs):
    """

    :param config:
    :return:
    """
    model_paths_list = config.model_paths_list
    model_names_list = config.model_names_list
    model_dims_list = config.model_dims_list
    char_model_path = config.char_model_path
    char_model_dims = config.char_model_dims

    if char_model_path:
        char_model = embedding_utils.reload_char2vec_model(char_model_path, char_model_dims)
    else:
        char_model = None

    embedding_models = embedding_utils.reload_embedding_models(model_paths_list,
                                                               model_names_list,
                                                               model_dims_list,
                                                               char_model)
    # doc_vector = embedding_models.get_vector_of_document(tokenized_text)
    return embedding_models


def get_emb_dim(emb_file):
    idx = 0
    dim = 0
    with open(emb_file, "r") as reader:
        if idx == 0:
            line = reader.readline().rstrip()
            dim = int(line.split(" ")[1])
    return dim


def extract_embedding_for_vocab_file(paths_of_emb_models, vocab_words_file, c2v_emb_file, output_file, output_format):
    """

    :param paths_of_emb_models:
    :param vocab_words_file:
    :param c2v_emb_file:
    :param output_file:
    :param output_format:
    :return:
    """
    config = EmbeddingConfigs()
    config.output_format = output_format
    config.model_paths_list = paths_of_emb_models.split(";")
    embedding_file_names = []
    embedding_dims = []

    if c2v_emb_file:
        config.char_model_path = c2v_emb_file
        config.char_model_dims = get_emb_dim(c2v_emb_file)

    print("02. Extracting word embeddings ...")
    if paths_of_emb_models and paths_of_emb_models.__contains__(";"):
        files = paths_of_emb_models.split(";")
        for emb_file in files:
            embedding_name = os.path.basename(os.path.normpath(emb_file))
            embedding_file_names.append(embedding_name)
            embedding_dim = get_emb_dim(emb_file)
            embedding_dims.append(embedding_dim)
    elif paths_of_emb_models: # In case there is only one embedding
        embedding_name = os.path.basename(os.path.normpath(paths_of_emb_models))
        embedding_file_names.append(embedding_name)
        embedding_dim = get_emb_dim(paths_of_emb_models)
        embedding_dims.append(embedding_dim)
    else:
        raise Exception("List of embeddings cannot be None.")

    # Data type:
    embedding_names = ["word2vec"]*len(embedding_dims) # embedding type, only support w2v and c2v type now
    config.model_names_list = embedding_names
    config.model_dims_list = embedding_dims

    # Do extracting embeddings
    extract_embedding_vectors(vocab_words_file, output_file, config)
    print("Done")


def extract_embedding_vectors(vocab_words_file, output_file, config: EmbeddingConfigs):
    """

    :param vocab_words_file:
    :param output_file:
    :param config:
    :return:
    """
    # Load vocab
    with Path(vocab_words_file).open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Output writer
    fwriter = open(output_file, "w")

    # Array of zeros
    dim_size = sum(config.model_dims_list)
    found = 0
    print('Reading embedding file (may take a while)')

    embedding_models = get_multi_embedding_models(config)

    embeddings = np.zeros((size_vocab, dim_size))

    line_idx = 0
    for word in word_to_idx.keys():
        word_idx = word_to_idx[word]

        word = word.rstrip()
        try:
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))

            w2v_vector = embedding_models.get_word_vector_of_multi_embeddings(word)

            if w2v_vector is not None and len(w2v_vector) > 0:
                embeddings[word_idx] = w2v_vector
                line = "%s %s" % (word, " ".join(str(scalar) for scalar in w2v_vector))
                fwriter.write(line + "\n")
                fwriter.flush()
                found += 1

            logging.debug("Embedding: ", w2v_vector)
        except Exception as e:
            logging.debug("Unexpected error: word = %s, error = %s" % (word, e))
            pass
        line_idx += 1

    print('- done. Found {} vectors for {} words'.format(found, size_vocab))
    fwriter.close()

    # Open file again to add meta data:
    src = open(output_file, "r")
    meta_line = "%s %s\n"%(found, dim_size)
    oline = src.readlines()
    # Here, we prepend the string we want to on first line
    oline.insert(0, meta_line)
    src.close()

    # We again open the file in WRITE mode
    src = open(output_file, "w")
    src.writelines(oline)
    src.close()
    # Done with writing.

    if config.output_format.__contains__(".gz"):
        content = open(output_file, "rb").read()
        gzip_out_file = output_file + '.gz'
        with gzip.open(gzip_out_file, 'wb') as f:
            f.write(content)
        print("Saved embedding to %s" % (gzip_out_file))

    if config.output_format.__contains__(".npz"):
        npz_out_file = output_file + '.npz'
        np.savez_compressed(npz_out_file, embeddings=embeddings)
        print("Saved embedding to %s"%(npz_out_file))
    return

