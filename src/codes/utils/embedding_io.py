from typing import Iterable, List, Set

from itertools import groupby
import numpy as np
import re
import utils.vectors as v
from utils.word import Word
import logging
import os
from embeddings.embedding_configs import EmbeddingConfigs


def save_model_to_file(embedding_model: List[Word], model_file_out: str):
    """
    Save loaded model back to file (to remove duplicated items).
    :param embedding_model:
    :param model_file_out:
    :return:
    """
    fwriter = open(model_file_out, "w")

    meta_data = "%s %s\n"%(len(embedding_model), len(embedding_model[0].vector))
    fwriter.write(meta_data)
    fwriter.flush()
    for w_Word in embedding_model:
        line = w_Word.text + " " + " ".join(str(scalar) for scalar in w_Word.vector.tolist())
        fwriter.write(line + "\n")
        fwriter.flush()
    fwriter.close()


def load_word_embeddings(file_paths: str, emb_config: EmbeddingConfigs) -> List[List[Word]]:
    """
    Sonvx: load multiple embeddings: e.g., <emb_file1>;<emb_file2>
    :param file_paths:
    :param emb_config:
    :return:
    """
    embedding_models = []
    embedding_names = []
    if file_paths and file_paths.__contains__(";"):
        files = file_paths.split(";")
        for emb_file in files:
            word_embedding = load_word_embedding(emb_file.replace("\"", ""), emb_config)
            embedding_name = os.path.basename(os.path.normpath(emb_file))
            embedding_models.append(word_embedding)
            embedding_names.append(embedding_name)
    else:
        return [load_word_embedding(file_paths), emb_config]

    return embedding_names, embedding_models


def load_word_embedding(file_path: str, emb_config: EmbeddingConfigs) -> List[Word]:
    """
    Load and cleanup the data.
    :param file_path:
    :param emb_config:
    :return:
    """
    print(f"Loading {file_path}...")
    words = load_words_raw(file_path, emb_config)
    print(f"Loaded {len(words)} words.")

    # Test
    word1 = words[1]
    print("Vec Len(word1) = ", len(word1.vector))

    # num_dimensions = most_common_dimension(words)
    # words = [w for w in words if len(w.vector) == dims]
    # print(f"Using {num_dimensions}-dimensional vectors, {len(words)} remain.")

    # words = remove_stop_words(words)
    # print(f"Removed stop words, {len(words)} remain.")

    # ords = remove_duplicates(words)
    # print(f"Removed duplicates, {len(words)} remain.")

    logging.debug("Embedding words: ", words[:10])
    print("Emb_vocab_size = ", len(words))
    # input("Done loading embedding: >>>>")

    return words


def load_words_raw(file_path: str, emb_config: EmbeddingConfigs) -> List[Word]:
    """
    Load the file as-is, without doing any validation or cleanup.
    :param file_path:
    :param emb_config:
    :return:
    """

    def parse_line(line: str, frequency: int) -> Word:
        # print("Line=", line)
        tokens = line.split(" ")
        word = tokens[0]
        if emb_config.do_normalize_emb:
            vector = v.normalize(np.array([float(x) for x in tokens[1:]]))
        else:
            vector = np.array([float(x) for x in tokens[1:]])
        return Word(word, vector, frequency)

    # Sonvx: NOT loading the same word twice.

    unique_dict = {}

    words = []
    # Words are sorted from the most common to the least common ones
    frequency = 1

    duplicated_entry = 0

    idx_counter, vocab_size, emb_dim = 0, 0, 0
    with open(file_path) as f:
        for line in f:
            line = line.rstrip()

            # print("Processing line: ", line)

            if idx_counter == 0 and emb_config.is_word2vec_format:
                try:
                    meta_info = line.split(" ")
                    vocab_size = int(meta_info[0])
                    emb_dim = int(meta_info[1])
                    idx_counter += 1
                    continue
                except Exception as e:
                    print("meta_info = "%(meta_info))
                    logging.error("Input embedding has format issue: Error = %s" % (e))

            # if len(line) < 20: # Ignore the first line of w2v format.
            #     continue

            w = parse_line(line, frequency)

            # Svx: only load if the word is not existed in the list.
            if w.text not in unique_dict:
                unique_dict[w.text] = frequency
                words.append(w)
                frequency += 1
            else:
                duplicated_entry += 1
                # print("Loading the same word again")

            # # Svx: check if the embedding dim is the same with the metadata, random check only
            if idx_counter == 10:
                if len(w.vector) != emb_dim:
                    message = "Metadata and the real vector size do not match: meta:real = %s:%s" \
                                  % (emb_dim, len(w.vector))
                    logging.error(message)
                    raise ValueError(message)
            idx_counter += 1

    if duplicated_entry > 0:
        logging.debug("Loading the same word again: %s"%(duplicated_entry))

    # Final check:
    if (frequency - 1) != vocab_size:
        msg = "Loaded %s/%s unique vocab." % ((frequency - 1), vocab_size)
        logging.info(msg)

    return words


def iter_len(iter: Iterable[complex]) -> int:
    return sum(1 for _ in iter)


def most_common_dimension(words: List[Word]) -> int:
    """
    There is a line in the input file which is missing a word
    (search -0.0739, -0.135, 0.0584).
    """
    lengths = sorted([len(word.vector) for word in words])
    dimensions = [(k, iter_len(v)) for k, v in groupby(lengths)]
    print("Dimensions:")
    for (dim, num_vectors) in dimensions:
        print(f"{num_vectors} {dim}-dimensional vectors")
    most_common = sorted(dimensions, key=lambda t: t[1], reverse=True)[0]
    return most_common[0]


# We want to ignore these characters,
# so that e.g. "U.S.", "U.S", "US_" and "US" are the same word.
ignore_char_regex = re.compile("[\W_]")

# Has to start and end with an alphanumeric character
is_valid_word = re.compile("^[^\W_].*[^\W_]$")


def remove_duplicates(words: List[Word]) -> List[Word]:
    seen_words: Set[str] = set()
    unique_words: List[Word] = []
    for w in words:
        canonical = ignore_char_regex.sub("", w.text)
        if not canonical in seen_words:
            seen_words.add(canonical)
            # Keep the original ordering
            unique_words.append(w)
    return unique_words


def remove_stop_words(words: List[Word]) -> List[Word]:
    return [w for w in words if (
            len(w.text) > 1 and is_valid_word.match(w.text))]


# Run "smoke tests" on import
assert [w.text for w in remove_stop_words([
    Word('a', [], 1),
    Word('ab', [], 1),
    Word('-ab', [], 1),
    Word('ab_', [], 1),
    Word('a.', [], 1),
    Word('.a', [], 1),
    Word('ab', [], 1),
])] == ['ab', 'ab']
assert [w.text for w in remove_duplicates([
    Word('a.b', [], 1),
    Word('-a-b', [], 1),
    Word('ab_+', [], 1),
    Word('.abc...', [], 1),
])] == ['a.b', '.abc...']
