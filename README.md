ETNLP: A Toolkit for Extraction, Evaluation and Visualization of Pre-trained Word Embeddings
=====

# Table of contents
1. [Introduction](#introduction)
2. [More about ETNLP](#moreaboutETNLP)
3. [Installation and How to Use](#installation_and_howtouse)
4. [Download Resources](#Download_Resources)


# I. Overview <a name="introduction"></a>
## A glimpse of ETNLP:
- Github: https://github.com/vietnlp/etnlp
- Video: https://vimeo.com/317599106
- Paper: https://arxiv.org/abs/1903.04433


# II. How do I cite ETNLP?
Please CITE paper the Arxiv paper whenever ETNLP (or the pre-trained embeddings) is used to produce published results or incorporated into other software:

```
@inproceedings{vu:2019n,
  title={ETNLP: A Visual-Aided Systematic Approach to Select Pre-Trained Embeddings for a Downstream Task},
  author={Vu, Xuan-Son and Vu, Thanh and Tran, Son N and Jiang, Lili},
  booktitle={Proceedings of the International Conference Recent Advances in Natural Language Processing (RANLP)},
  year={2019}
  }
```


# III. More about ETNLP <a name="moreaboutETNLP"></a>:
## 1. Embedding Evaluator: 
To compare quality of embedding models on the word analogy task.
- Input: a pre-trained embedding vector file (word2vec format), and word analogy file.
- Output: (1) evaluate quality of the embedding model based on the MAP/P@10 score, (2) Paired t-tests to show significant level between different word embeddings.

### 1.1. Note: The word analogy list is created by:
- Adopt from the English list by selecting suitable categories and translating to the target language (i.e., Vietnamese). 
- Removing inappropriate categories (i.e., category 6, 10, 11, 14) in the target language (i.e., Vietnamese).
- Adding custom category that is suitable for the target language (e.g., cities and their zones in Vietnam for Vietnamese).
Since most of this process is automatically done, it can be applied in other languages as well.

### 1.2. Selected categories for Vietnamese:  
> 1. capital-common-countries
> 2. capital-world
> 3. currency: E.g., Algeria | dinar | Angola | kwanza
> 4. city-in-zone (Vietnam's cities and its zone)
> 5. family (boy|girl | brother | sister)
> 6. gram1-adjective-to-adverb (NOT USED)
> 7. gram2-opposite (e.g., acceptable | unacceptable | aware | unaware)
> 8. gram3-comparative (e.g., bad | worse | big | bigger)
> 9. gram4-superlative (e.g., bad | worst | big | biggest)
> 10. gram5-present-participle (NOT USED)
> 11. gram6-nationality-adjective-nguoi-tieng (e.g., Albania | Albanian | Argentina | Argentinean)
> 12. gram7-past-tense (NOT USED)
> 13. gram8-plural-cac-nhung (e.g., banana | bananas | bird | birds) (NOT USED)
> 14. gram9-plural-verbs (NOT USED)

### 1.3 Evaluation results (in details)

* Analogy: Word Analogy Task

* NER (w): NER task with hyper-parameters selected from the best F1 on validation set.

* NER (w.o): NER task without selecting hyper-parameters from the validation set.

| ﻿            Model            | NER.w        | NER.w.o 	| Analogy 	|
|------------------------------	|------------- | ------------------	|------------------	|
| BiLC3 + w2v                 	| 89.01   | 89.41            	|         0.4796 |
| BiLC3 + Bert_Base           	| 88.26   | 89.91             | 0.4609 |
| BiLC3 + w2v_c2v             	| 89.46   | 89.46            	| 0.4796 |
| BiLC3 + fastText            	| 89.65   | 89.84            	|  0.4970 |
| BiLC3 + Elmo                	| 89.67 | 90.84            	| **0.4999** |
| BiLC3 + MULTI_WC_F_E_B        | **91.09**   | **91.75**        	| 0.4906|


## 2. Embedding Extractor: To extract embedding vectors for other tasks.
- Input: (1) list of input embeddings, (2) a vocabulary file.
- Output: embedding vectors of the given vocab file in `.txt`, i.e., each line conains the embedding for a word. The file then be compressed in .gz format. This format is widely used in existing NLP Toolkits (e.g., Reimers et al. [1]).

### Extra options:
- `-input-c2v`: character embedding file
- `solveoov:1`: to solve OOV words of the 1st embedding. Similarly for more than one embedding: e.g., `solveoov:1:2`.

[1] Nils Reimers and Iryna Gurevych, Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging, 2017, http://arxiv.org/abs/1707.09861, arXiv.

## 3. Visualizer: to explore the embedding space and compare between different embeddings.

### Screenshot of viewing multiple-embeddings side-by-side (Vietnamese):
![Alt text](images/etnlp_view_multi_embeddings.png "Screenshot multiple-embeddings side-by-side")

### Screenshot of viewing each embedding interactively (Vietnamese):
![Alt text](images/etnlp_view_embs.png "Screenshot example of viewing each embedding interactively")

### Screenshot of viewing each embedding side-by-side (English):
![Alt text](images/english_sbs.png "Screenshot example of viewing each embedding interactively")


# IV. Installation and How to use ETNLP <a name="installation_and_howtouse"></a>
## 1. Installation:

From source codes (Python 3.6.x):
> 1. cd src/codes/
> 2. pip install -r requirements.txt
> 3. python setup.py install

From pip (python 3.6.x)
> 1. sudo apt-get install python3-dev
> 2. pip install cython
> 3. pip install git+git://github.com/vietnlp/etnlp.git

OR:
> 1. pip install etnlp 

## 2. Examples
> 1. cd src/examples
> 2. python test1_etnlp_preprocessing.py
> 3. python test2_etnlp_extractor.py
> 4. python test3_etnlp_evaluator.py
> 5. python test4_etnlp_visualizer.py
### Example of using Fasttext-Sent2Vec:
- 01. Install: https://github.com/epfml/sent2vec
```
01. git clone https://github.com/epfml/sent2vec
02. cd sent2vec; pip install .
```

- 02. Extract embeddings for sentences (no requirement for tokenization before extracting embedding of sentences).
```
import sent2vec
model = sent2vec.Sent2vecModel()
model.load_model('opendata_wiki_lowercase_words.bin')
emb = model.embed_sentence("tôi là sinh viên đh công nghệ, đại học quôc gia hà nội") 
embs = model.embed_sentences(["tôi là sinh viên", "tôi là nhà thơ", "tôi là bác sĩ"])

```


### 3. Visualization
Side-by-side visualization:
> 1. sh src/codes/04.run_etnlp_visualizer_sbs.sh

Interactive visualization:
> 1. sh src/codes/04.run_etnlp_visualizer_inter.sh


# V. Available Lexical Resources <a name="Download_Resources"></a>
## 1. Word Analogy List for Vietnamese

| ﻿ Word Analogy List          | Download Link (NER Task)| Download Link (General)| 
|------------------------------|---------------|---------------|
| Vietnamese (This work)       | [Link1](https://drive.google.com/file/d/1eA5yvla4BhAIfWsmZherT1GEW6gzDC-1/view?usp=sharing)| [Link1](https://drive.google.com/file/d/1YJ9d5rVKMMKF1xWWZi26_sNpgULTvxwg/view?usp=sharing)|
| English (Mirkolov et al. [2]) | [Link2]| [Link2](https://drive.google.com/file/d/10rWxGu8-nbQmYC8wrIussSZjY0lDh6RP/view?usp=sharing)|
| Portuguese (Hartmann et al. [3]) | [Link3]| [Link3](https://github.com/nathanshartmann/portuguese_word_embeddings/blob/master/analogies/testset/LX-4WAnalogies.txt)|



## 2. Multiple pre-trained embedding models for Vietnamese

- Training data: Wiki in Vietnamese:

| ﻿ # of sentences          | # of tokenized words| 
|------------------------------|---------------|
| ﻿  6,685,621          | 114,997,587 |


- Download Pre-trained Embeddings: <br>
(Note: The MULTI_WC_F_E_B is the concatenation of four embeddings: W2V_C2V, fastText, ELMO, and Bert_Base.)

| ﻿  Embedding Model           | Download Link (NER Task) | Download Link (AIVIVN SentiTask) | Download Link (General) | 
|------------------------------|---------------|---------------|---------------|
| w2v                          | [Link1](https://drive.google.com/file/d/1LHaZ8LXxteHzod42naqJZYCwwq5mI9aL/view?usp=sharing) (dim=300)|  [Link1] | [Link1] |
| w2v_c2v                      | [Link2](https://drive.google.com/file/d/1-M9Tb9l8mNmP3RKxZiZNK1Vpbng2yw4l/view?usp=sharing) (dim=300)|  [Link2] | [Link2] |
| fastText                     | [Link3](https://drive.google.com/file/d/1dHCPhKFjtDjbrUeeymheDnlhjtaljPGE/view?usp=sharing) (dim=300)|  [Link3] | [Link3] |
| fastText-[Sent2Vec](https://github.com/epfml/sent2vec)                     | [Link3]|  [Link3] | [Link3](https://drive.google.com/file/d/1BzL1mpdfqCCJioCdAlTVshbrz0lGfP2D/view?usp=sharing) (dim=300, 6GB, trained on 20GB of [news data](https://github.com/binhvq/news-corpus) and Wiki-data of ETNLP. |
| Elmo                         | [Link4](https://drive.google.com/file/d/1zDaSD8NsZNXGyd9iVOxTcb7CP61Ixo-r/view?usp=sharing) (dim=1024)|  [Link4](https://drive.google.com/file/d/1jVJtF0f6SbtUd-t3bnywP6mFnz0QXPIx/view?usp=sharing) (dim=1024)| [Link4](https://drive.google.com/file/d/1XPsTzg1Gex-Hh2nl9344YlZc1orOVBDp/view?usp=sharing) (dim=1024, 731MB and 1.9GB after extraction.)|
| Bert_base                    | [Link5](https://drive.google.com/file/d/16fRkmIHiB16OlM8WdFmoApGtLMf6YJJ8/view?usp=sharing) (dim=768)|  [Link5] | [Link5] |
| MULTI_WC_F_E_B               | [Link6](https://drive.google.com/file/d/1gq7b8hs31VzoeO3n3C__ftlDnE_iBZW2/view?usp=sharing)  (dim=2392)| [Link6] | [Link6] |


# VI. Versioning
For transparency and insight into our release cycle, and for striving to maintain backward compatibility, ETNLP will be maintained under the Semantic Versioning guidelines as much as possible.

Releases will be numbered with the following format:

`<major>.<minor>.<patch>`

And constructed with the following guidelines:

* Breaking backward compatibility bumps the major (and resets the minor and patch)
* New additions without breaking backward compatibility bumps the minor (and resets the patch)
* Bug fixes and misc changes bumps the patch

For more information on SemVer, please visit http://semver.org/.
