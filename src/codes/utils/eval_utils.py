
"""
MAP@K word level and character level are explained in detail in this paper:

dpUGC: Learn Differentially Private Representationfor User Generated Contents
Xuan-Son Vu, Son N. Tran, Lili Jiang
In: Proceedings of the 20th International Conference on Computational Linguistics and
Intelligent Text Processing, April, 2019, (to appear)

Please cite the above paper if you use codes in this file.

"""


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10, word_level=True):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    # print("Sending arr = ", arr)
    if word_level:
        return calc_map(actual, predicted, topK=k)
    else:
        # arr = [apk(a, p, k) for a, p in zip(actual, predicted)]
        # return np.mean(arr)
        return calc_map_character_level(actual, predicted, topK=k)


def calc_map(actual, predicted, topK=10):
    """

    :param actual:
    :param predicted:
    :param topK:
    :return:
    """
    # print("Input: actual %s, predicted %s"%(actual, predicted))
    if len(predicted) > topK:
        predicted = predicted[:topK]
    idx = 1
    hit = 0
    map_arr = []
    for answer in predicted:
        if answer in actual[:topK]:
            hit += 1
            val = (hit * 1.0) / (idx * 1.0)
            # print("hit = %s, idx = %s"%(hit, idx))
            map_arr.append(val)
            # print("hit: %s, map_arr = %s"%(answer, map_arr))
        idx += 1
    # print("map_arr = %s done", map_arr)
    if len(map_arr) > 0:
        return np.mean(map_arr)
    else:
        return 0.0


def calc_map_character_level(actual, predicted, topK=10):
    """

    :param actual:
    :param predicted:
    :param topK:
    :return:
    """
    # print("Input: actual %s, predicted %s" % (actual, predicted))
    if len(predicted) > topK:
        predicted = predicted[:topK]

    if len(actual) > topK:
        actual = actual[:topK]

    rank = 1
    hit = 0
    actual_seq = ''.join([word for word in actual])
    predicted_seq = ''.join([word for word in predicted])
    map_arr = []
    for char in predicted_seq:
        if char in actual_seq[:rank]:
            hit += 1
            val = (hit * 1.0) / (rank * 1.0)
            # print("hit = %s, idx = %s" % (hit, rank))
            map_arr.append(val)
            # print("hit: %s, map_arr = %s" % (char, map_arr))
        rank += 1
    # print("map_arr = %s done", map_arr)
    return np.mean(map_arr)


import unittest
import numpy as np


def test_apk(self):
    self.assertAlmostEqual(apk(range(1, 6), [6, 4, 7, 1, 2], 2), 0.25)
    self.assertAlmostEqual(apk(range(1, 6), [1, 1, 1, 1, 1], 5), 0.2)
    predicted = range(1, 21)
    predicted.extend(range(200, 600))
    self.assertAlmostEqual(apk(range(1, 100), predicted, 20), 1.0)


def test_mapk(self):
    self.assertAlmostEqual(mapk([range(1, 5)], [range(1, 5)], 3), 1.0)
    self.assertAlmostEqual(mapk([[1, 3, 4], [1, 2, 4], [1, 3]],
                                [range(1, 6), range(1, 6), range(1, 6)], 3), 0.685185185185185)
    self.assertAlmostEqual(mapk([range(1, 6), range(1, 6)],
                                [[6, 4, 7, 1, 2], [1, 1, 1, 1, 1]], 5), 0.26)
    self.assertAlmostEqual(mapk([[1, 3], [1, 2, 3], [1, 2, 3]],
                                [range(1, 6), [1, 1, 1], [1, 2, 1]], 3), 11.0 / 18)


if __name__ == '__main__':
    a1 = ["1", '2', '3', '4']
    b1 = ['1', '5', '2', '8']
    print(mapk(a1, b1, 4))

    a1 = ["15"]
    b1 = ["1", "2", "3", "4", "5","6","7","8","9","10"]

    print("MapK:", mapk(a1, b1, 4))

    # unittest.main()
