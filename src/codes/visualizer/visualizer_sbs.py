from flask import Flask, render_template
from flask import request
import gensim
from distutils.version import LooseVersion
from utils import string_utils
import sys


app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

global embedding_models


@app.route('/search', methods=['GET', 'POST'])
def search():
    """
    Get input query and return list of top similiar words in all embeddings.
    :param embedding_paths_arr:
    :return:
    """
    if request.method == "POST":
        query = request.values['search'] or ''
        # query = unicode(query, "utf-8")
        # query = query.decode().encode("utf-8")
        # Python 2.7
        try:
            # Old
            # query = unicode(query).lower()
            query = string_utils.convert_to_unicode(query)
        except Exception as e:
            raise Exception("Something went wrong: msg = %s, query = %s."%(e, query))

        print('query = ' + query)
        output_arr = []

        for embedding_model in embedding_models:
            try:
                output = []
                sim_list = embedding_model.most_similar(query, topn=50)
                for wordsimilar in sim_list:
                    output.append(wordsimilar[0] + ' - ' + str(round(wordsimilar[1], 6)))

                output_arr.append(output)
            except Exception as e:
                output = 'Err: %s, Not found query = %s' % (e, query)
                output_arr.append(output)

    return render_template('search.html',
                           embedding_names_arr=embedding_names_arr,
                           output_arr=output_arr
                           )


@app.route("/")
def get_index():
    return render_template('search.html')


@app.route("/multi_search")
def multi_search():
    return render_template('multi_search.html')


if __name__ == "__main__":
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    # download pre-trained_models at https://github.com/vietnlp/etnlp
    if len(sys.argv) < 2:
        print("Missing input arguments. Input format: ./*.py <emb_file1;emb_file2;...>. Exiting ...")
        exit(0)

    if sys.argv[1].__contains__(";"):
        model_files = sys.argv[1].split(";")
    else:
        model_files = [sys.argv[1]]

    embedding_names_arr = [os.path.basename(file_path) for file_path in model_files]

    embedding_models = []
    idx = 0
    for model in model_files:
        # model = root_dir + model
        if os.path.isfile(model):
            print('Loading embedding model ... %s' % (idx))

            isBinary = False
            if model.endswith(".bin"):
                isBinary = True

            if LooseVersion(gensim.__version__) >= LooseVersion("1.0.1"):
                from gensim.models import KeyedVectors

                embedding_models.append(KeyedVectors.load_word2vec_format(model, binary=isBinary))
            else:
                from gensim.models import Word2Vec

                embedding_models.append(Word2Vec.load_word2vec_format(model, binary=isBinary))
            idx += 1
        else:
            print(
                "Download word2vec model and put into ../data/. File: https://github.com/vietnlp/etnlp")

    app.run(debug=False, port=8089, host='0.0.0.0')
