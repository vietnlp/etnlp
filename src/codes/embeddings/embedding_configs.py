

class EmbeddingConfigs(object):
    """
        Configuration information
    """
    is_word2vec_format = True
    do_normalize_emb = True
    model_paths_list = []
    model_names_list = []
    model_dims_list = []
    char_model_path = None
    char_model_dims = -1
    output_format = ".txt;.npz;.gz"
