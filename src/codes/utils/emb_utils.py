from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Iterable, List, Optional, Set, Tuple

from utils.vectors import Vector
from utils import vectors
from utils.word import Word
from utils import eval_utils
from gensim import utils as genutils
import logging
import numpy as np
from scipy import stats

# Timing info for most_similar (100k words):
# Original version: 7.3s
# Normalized vectors: 3.4s
logger = logging.getLogger(__name__)


def most_similar(base_vector: Vector, words: List[Word]) -> List[Tuple[float, Word]]:
    """Finds n words with smallest cosine similarity to a given word"""
    words_with_distance = [(vectors.cosine_similarity_normalized(base_vector, w.vector), w) for w in words]
    # We want cosine similarity to be as large as possible (close to 1)
    sorted_by_distance = sorted(words_with_distance, key=lambda t: t[0], reverse=True)
    # Sonvx: remove duplications (not understand why yet, probably because the w2v?)
    # sorted_by_distance = list(set(sorted_by_distance))
    return sorted_by_distance


def print_most_similar(words: List[Word], text: str) -> None:
    base_word = find_word(text, words)
    if not base_word:
        print(f"Unknown word: {text}")
        return
    print(f"Words related to {base_word.text}:")
    sorted_by_distance = [
        word.text for (dist, word) in
        most_similar(base_word.vector, words)
        if word.text.lower() != base_word.text.lower()
    ]
    print(', '.join(sorted_by_distance[:10]))


def read_word() -> str:
    return input("Type a word: ")


def find_word(text: str, words: List[Word]) -> Optional[Word]:
    try:
        return next(w for w in words if text == w.text)
    except StopIteration:
        return None


def closest_analogies_OLD(
        left2: str, left1: str, right2: str, words: List[Word]
) -> List[Tuple[float, Word]]:
    word_left1 = find_word(left1, words)
    word_left2 = find_word(left2, words)
    word_right2 = find_word(right2, words)
    if (not word_left1) or (not word_left2) or (not word_right2):
        return []
    vector = vectors.add(
        vectors.sub(word_left1.vector, word_left2.vector),
        word_right2.vector)
    closest = most_similar(vector, words)[:10]

    def is_redundant(word: str) -> bool:
        """
        Sometimes the two left vectors are so close the answer is e.g.
        "shirt-clothing is like phone-phones". Skip 'phones' and get the next
        suggestion, which might be more interesting.
        """
        word_lower = word.lower()
        return (
                left1.lower() in word_lower or
                left2.lower() in word_lower or
                right2.lower() in word_lower)

    closest_filtered = [(dist, w) for (dist, w) in closest if not is_redundant(w.text)]
    return closest_filtered


def closest_analogies_vectors(
        word_left2: Word, word_left1: Word, word_right2: Word, words: List[Word]) \
            -> List[Tuple[float, Word]]:
    """
    Sonvx:
    :param word_left2:
    :param word_left1:
    :param word_right2:
    :param words:
    :param remove_redundancy: remove suggestions if they contain the given words.
    :return:
    """
    # print(">>>> Remove redundancy = ", remove_redundancy)
    # input(">>>>")
    vector = vectors.add(
        vectors.sub(word_left1.vector, word_left2.vector),
        word_right2.vector)
    closest = most_similar(vector, words)[:10]

    def is_redundant(word: str) -> bool:
        """
        Sometimes the two left vectors are so close the answer is e.g.
        "shirt-clothing is like phone-phones". Skip 'phones' and get the next
        suggestion, which might be more interesting.
        """
        word_lower = word.lower()
        return (
                word_left1.text.lower() in word_lower or
                word_left2.text.lower() in word_lower or
                word_right2.text.lower() in word_lower)
    # It doesn't work this way for Vietnamese, so we try both of this to test for now
    if False:
        closest_filtered = [(dist, w) for (dist, w) in closest if not is_redundant(w.text)]
    else:
        closest_filtered = closest
    return closest_filtered


def get_avg_vector(word, embedding_words):

    if " " in word:
        single_words = word.split(" ")
        list_vector = []

        for single_word in single_words:
            word_vec = find_word(single_word, embedding_words)
            if word_vec:
                list_vector.append(word_vec.vector)
            else:
                # Try again with lowercase
                single_word = single_word.lower()
                word_vec = find_word(single_word, embedding_words)
                if word_vec:
                    list_vector.append(word_vec.vector)

        # print("list_vector: ", list_vector)
        # input(">>>>>>>>")

        returned_Word = Word(word, vectors.mean_list(list_vector), 1)
    else:
        returned_Word = find_word(word, embedding_words)

    # print("Avg returned vector = ", returned_vector)
    # input(">>>>")

    return returned_Word


def run_paired_ttests(all_map_arr, embedding_names):
    """
    Run Paired t-tests on MAP results
    :param all_map_arr:
    :param embedding_names:
    :return:
    """
    str_out = ""
    num_embs = len(all_map_arr)

    # Verify to make sure they have the same length
    if all_map_arr and embedding_names:
        for i in range(0, num_embs - 1):
            for j in range(i + 1, num_embs):
                if len(all_map_arr[i]) != len(all_map_arr[j]):
                    raise Exception("Two embedding (%s, %s) have different MAP list, sizes: %s vs. %s"
                                    % (embedding_names[i], embedding_names[j], len(all_map_arr[i]), len(all_map_arr[j])))
    else:
        logging.error("Inputs are NULL")

    result_str_ttest_arr = []
    for i in range(0, num_embs - 1):
        for j in range(i + 1, num_embs):
            stat_test_ret = stats.ttest_rel(all_map_arr[i], all_map_arr[j])
            # if stat_test_ret.pvalue >= 0.05:
            result = "%s vs. %s: %s" % (embedding_names[i], embedding_names[j], stat_test_ret)
            str_out += result + "\n"

    return str_out


def eval_word_analogy_4_all_embeddings(word_analogies_file, embedding_names: List[str],
                                       word_embeddings: List[List[Word]], output_file):
    """
    Run word analogy for all embeddings
    :param word_analogies_file:
    :param embedding_names:
    :param word_embeddings:
    :param output_file:
    :return:
    """
    fwriter = open(output_file, "w")
    idx = 0
    all_map_arr = []
    console_output_str = ""
    category = ": | Word Analogy Task results\n"
    fwriter.write(category)
    console_output_str += category

    for word_embedding in word_embeddings:
        embedding_name = embedding_names[idx]
        map_at_10, map_arr, result_str = eval_word_analogies(word_analogies_file, word_embedding, embedding_name)
        all_map_arr.append(map_arr)
        meta_info = "\nEmbedding: %s"%(embedding_names[idx])
        fwriter.write(meta_info + "\n")
        fwriter.write(result_str)
        fwriter.write("MAP_arr = %s"%(map_arr))
        fwriter.write("MAP@10 = %s" % (map_at_10))
        fwriter.flush()
        console_output_str += meta_info + "\n" + "MAP@10 = %s" % (map_at_10) + "\n"
        idx += 1

    # Getting significant Paired t-tests
    category = "\n: | Paired t-tests results\n"
    fwriter.write(category)
    console_output_str += category
    ttests_result = run_paired_ttests(all_map_arr, embedding_names)
    console_output_str += ttests_result
    fwriter.write(ttests_result)
    fwriter.flush()
    fwriter.close()

    return console_output_str


def eval_word_analogies(word_analogies_file, words: List[Word], embedding_name):
    """
    Sonvx: Evaluate word analogy for one embedding.
    :param word_analogies_file:
    :param words:
    :return:
    """
    # input("GO checking >>>>")
    oov_counter, idx_cnt, is_vn_counter, phrase_cnt = 0, -1, 0, 0
    sections, section = [], None
    # map_arr = []
    out_str = ""
    map_ret_dict = {}

    for line_no, line in enumerate(genutils.smart_open(word_analogies_file)):
        # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
        line = genutils.to_unicode(line)
        line = line.rstrip()
        if line.startswith(': |'):
            # a new section starts => store the old section
            if section:
                sections.append(section)
            section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
        else:
            # Count number of analogy to check
            idx_cnt += 1

            # Set default map value
            map_ret_dict[idx_cnt] = 0.0

            if not section:
                raise ValueError("missing section header before line #%i in %s" % (line_no, word_analogies_file))
            try:
                # a - b + c = expected
                # Input: Baghdad | Irac | Bangkok | Thai_Lan
                # Baghdad - Irac = Bangkok - Thai_Lan
                # -> Baghdad - Irac + Thai_Lan = Bangkok
                # =>
                a, b, expected, c = [word for word in line.split(" | ")]
            except ValueError:
                logger.debug("SVX: ERROR skipping invalid line #%i in %s", line_no, word_analogies_file)
                print("Line : ", line)
                print("a, b, c, expected: %s, %s, %s, %s" % (a, b, c, expected))
                # input(">>> Wait ...")
                continue

            # In case of Vietnamese, word analogy can be a phrase
            if " " in expected:
                print("INFO: we don't support to find word analogies for phrase for NOW.")
                phrase_cnt += 1
                continue
            elif " " in a or " " in b or " " in c:
                is_vn_counter += 1
                word_left1 = get_avg_vector(a, words)
                word_left2 = get_avg_vector(b, words)
                word_right2 = get_avg_vector(c, words)
            else:
                word_left1 = find_word(a, words)
                word_left2 = find_word(b, words)
                word_right2 = find_word(c, words)

            if (not word_left1) or (not word_left2) or (not word_right2):
                logger.debug("SVX: skipping line #%i with OOV words: %s", line_no, line.strip())
                oov_counter += 1
                continue

            # Write solable analogy to a file
            # fsolveable_writer.write(line + "\n")

            logger.debug("word_left1 = %s", word_left1.text)
            logger.debug("word_left2 = %s", word_left2.text)
            logger.debug("word_right2 = %s", word_right2.text)

            # Start finding close word:
            # Note: we can only find 1 expected word in Vietnamese for NOW
            top10_candidate = closest_analogies_vectors(word_left2, word_left1,
                                                        word_right2, words)
            list_candidate_arr = []
            for tuple in top10_candidate:
                list_candidate_arr.append(tuple[1].text)

            logger.debug("Expected Word: %s, candidate = %s" % (expected, list_candidate_arr))
            # input(">>>>>")
            # Calculate MAP@10 score
            this_map_result = eval_utils.mapk(expected, list_candidate_arr, word_level=True)
            if this_map_result >= 0:
                this_map_result = round(this_map_result, 6)
                # map_arr[idx_cnt] = this_map_result
            else:
                this_map_result = 0.0
                # map_arr.append(0.0)
                # map_arr[idx_cnt] = this_map_result

            map_ret_dict[idx_cnt] = this_map_result

            if expected in list_candidate_arr:
                section['correct'].append((a, b, c, expected))
                out_line = "%s - %s + %s = ?; Expect: %s, candidate: %s" % \
                           (word_left1, word_left2, word_right2, expected, list_candidate_arr)
                out_str += out_line + "\n"

            # else:
            #    section['incorrect'].append((a, b, c, expected))

    # fsolveable_writer.close()
    if section:
        # store the last section, too
        sections.append(section)

    map_arr = list(map_ret_dict.values())
    logger.debug("map_arr = ", map_arr)
    logger.debug("MAP_RET_DICT = ", map_ret_dict)
    # input("Check result dict: >>>>>")

    total = {
        "Emb_Name: " + embedding_name + '/OOV/Total/VN_Solveable_Cases/VN_Phrase_Target':
            [oov_counter, (idx_cnt + 1), is_vn_counter, phrase_cnt],
        'MAP@10': np.mean(map_arr)
        # ,
        # 'section': 'total'
        # ,
        # 'correct': sum((s['correct'] for s in sections), []),
        # 'incorrect': sum((s['incorrect'] for s in sections), []),
    }
    # print (out_str)
    # print(total)
    # logger.info(total)

    sections.append(total)
    sections_str = "\n%s\n" % sections

    return np.mean(map_arr), map_arr, sections_str


def print_analogy(left2: str, left1: str, right2: str, words: List[Word]) -> None:
    analogies = closest_analogies_OLD(left2, left1, right2, words)
    if (len(analogies) == 0):
        print(f"{left2}-{left1} is like {right2}-?")
        # man-king is like woman-king
        # input: man is to king is like woman is to ___?(queen).
    else:
        (dist, w) = analogies[0]
        # alternatives = ', '.join([f"{w.text} ({dist})" for (dist, w) in analogies])
        print(f"{left2}-{left1} is like {right2}-{w.text}")

