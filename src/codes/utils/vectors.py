from typing import List, Any, Optional

import math
import numpy as np

# Adopt from https://github.com/mkonicek/nlp/vecters.py

# Vector = np.ndarray[float]
Vector = 'np.ndarray[float]'
vector_type = 'np.ndarray[float]'

# Vector = np.ndarray(dtype=float)


def l2_len(v: vector_type) -> float:
    return math.sqrt(np.dot(v, v))


def dot(v1: vector_type, v2: vector_type) -> float:
    assert v1.shape == v2.shape
    return np.dot(v1, v2)


def mean(v1: vector_type, v2: vector_type) -> Vector:
    """
    Added by Sonvx: get mean of 2 vectors.
    :param v1:
    :param v2:
    :return:
    """
    assert v1.shape == v2.shape
    return np.mean([v1, v2], axis=0)


def mean_list(v1: List[Vector]) -> Vector:
    """
    Added by Sonvx: get mean of 2 vectors.
    :param v1:
    :return:
    """
    if len(v1) > 0:
        return np.mean(v1, axis=0)
    else:
        return None


def add(v1: vector_type, v2: vector_type) -> Vector:
    assert v1.shape == v2.shape
    return np.add(v1, v2)


def sub(v1: vector_type, v2: vector_type) -> Vector:
    assert v1.shape == v2.shape
    return np.subtract(v1, v2)


def normalize(v: vector_type) -> Vector:
    return v / l2_len(v)


def cosine_similarity_normalized(v1: vector_type, v2: vector_type) -> float:
    """
    Returns the cosine of the angle between the two vectors.
    Each of the vectors must have length (L2-norm) equal to 1.
    Results range from -1 (very different) to 1 (very similar).
    """
    return dot(v1, v2)
