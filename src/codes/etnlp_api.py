import argparse
from api import embedding_preprocessing, embedding_evaluator, embedding_extractor, embedding_visualizer
import logging
import os
from embeddings.embedding_configs import EmbeddingConfigs


embedding_config = EmbeddingConfigs()

if __name__ == "__main__":
    """
    ETNLP: a toolkit for evaluate, extract, and visualize multiple word embeddings
    """

    _desc = "Evaluates a word embedding model"
    _parser = argparse.ArgumentParser(description=_desc)
    _parser.add_argument("-input",
                         required=True,
                         default="../data/embedding_dicts/elmo_embeddings.txt",
                         #
                         help="model")
    _parser.add_argument("-analoglist",
                         nargs="?",
                         # default="../data/embedding_analogies/vi/analogy_vn_seg.txt.std.txt",
                         default="./data/embedding_analogy/solveable_analogies_vi.txt",
                         help="testset")

    _parser.add_argument("-args",
                         nargs="?",
                         default="eval",
                         help="Run evaluation")

    _parser.add_argument("-lang",
                         nargs="?",
                         default="VI",
                         help="Specify language, by default, it's Vietnamese.")

    _parser.add_argument("-vocab",
                         nargs="?",
                         default="../data/vocab.txt",
                         help="Vocab to be extracted")

    _parser.add_argument("-input_c2v",
                         nargs="?",
                         default=None,
                         help="C2V embedding")

    _parser.add_argument("-output",
                         nargs="?",
                         default="../data/embedding_analogies/vi/results_out.txt",
                         help="Output file of word analogy task")

    _parser.add_argument("-output_format",
                         nargs="?",
                         default=".txt",
                         help="Format of output file of the extracted embedding.")

    _args = _parser.parse_args()

    # Set logging level
    logging.basicConfig(level=logging.INFO)
    logging.disable(logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

    input_embedding_files_str = _args.input
    analoglist = _args.analoglist
    is_vietnamese = _args.lang
    output_files_str = _args.output
    options_str = _args.args
    vocab_file = _args.vocab
    output_format = _args.output_format

    # By default, we process all embeddings as word2vec format.
    embedding_preprocessing.is_word2vec_format = True

    if options_str == 'eval':
        print("Starting evaluator ...")
        embedding_evaluator.evaluator_api(input_files=input_embedding_files_str, analoglist=analoglist,
                                          output=output_files_str)
        print("Done evaluator !")
    elif options_str == 'visualizer':
        print("Starting visualizer ...")
        embedding_visualizer.visualize_multiple_embeddings(input_embedding_files_str)
        print("Done visualizer !")
    elif options_str.startswith("extract"):
        print("Starting extractor ...")
        embedding_extractor.extract_embedding_for_vocab_file(input_embedding_files_str, vocab_file,
                                                             _args.input_c2v, output_files_str, output_format)
        print("Done extractor !")
    elif options_str.startswith("glove2w2v"):
        print("Starting pre-processing: convert to word2vec format ...")
        embedding_config.is_word2vec_format = False
        if options_str.__contains__("do_normalize"):
            embedding_config.do_normalize_emb = True
        else:
            embedding_config.do_normalize_emb = False
        embedding_preprocessing.load_and_save_2_word2vec_models(input_embedding_files_str,
                                                                output_files_str,
                                                                embedding_config)

    else:
        print("Invalid options")

    print("Done!")




