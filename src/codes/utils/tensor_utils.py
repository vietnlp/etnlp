import numpy as np
import tensorflow as tf


def fusion_tensor(emb_tensor1, emb_tensor2, emb_tensor3):
    """
    Fusion products between 3 tensors
    :param emb_tensor1:
    :param emb_tensor2:
    :param emb_tensor3:
    :return:
    """
    x1_dim = emb_tensor1.shape[2]  # shape (7, 10)
    x2_dim = emb_tensor2.shape[2]
    x3_dim = emb_tensor3.shape[2]

    fusion_W_dim = 3 + 2  # number of embedding + 2 more dims
    w = np.random.randn(x1_dim, x2_dim, x3_dim, fusion_W_dim)

    # t1_pre0 = tf.expand_dims(emb_tensor1, -1)
    # t1_pre1 = tf.expand_dims(t1_pre0, -1)
    # t1 = tf.tile(t1_pre1, [1, 1, x2_dim, x3_dim])
    # print("t1_pre0 = %s, t1_pre1 = %s, t1 = %s" % (t1_pre0.shape, t1_pre1.shape, t1.shape))
    # t1 = tf.tile(tf.expand_dims(tf.expand_dims(v1,-1),-1),[1,1,20,30])
    t1 = tf.tile(tf.expand_dims(tf.expand_dims(emb_tensor1, -1), -1), [1, 1, 1, x2_dim, x3_dim])
    t2 = tf.tile(tf.expand_dims(tf.expand_dims(emb_tensor2, 1), -1), [1, 1, x1_dim, 1, x3_dim])
    t3 = tf.tile(tf.expand_dims(tf.expand_dims(emb_tensor3, 1), 2), [1, 1, x1_dim, x2_dim, 1])
    tw = tf.convert_to_tensor(w, dtype=tf.float64)  # might change to tf.zeros_like?

    t123 = t1 * t2 * t3
    h = tf.tensordot(t123, tw, axes=[[1, 2, 3], [0, 1, 2]])
    return h
