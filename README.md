# 1. Embedding Evaluator: To compare quality of embedding models on the word analogy task.
- Input: a pre-trained embedding vector file (word2vec format), and word analogy file.
- Output: (1) evaluate quality of the embedding model based on the MAP/P@10 score, (2) Paired t-tests to show significant level between different word embeddings.

## 1.1. Note: The word analogy list is created by:
- Adopt from the English list by selecting suitable categories and translating to the target language (i.e., Vietnamese). 
- Removing inappropriate categories (i.e., category 6, 10, 11, 14) in the target language (i.e., Vietnamese).
- Adding custom category that is suitable for the target language (e.g., cities and their zones in Vietnam for Vietnamese).
Since most of this process is automatically done, it can be applied in other languages as well.

## 1.2. Selected categories for Vietnamese:  
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

# 2. Evaluation results (in details)

* Analogy: Word Analogy Task

* NER (w): NER task with hyper-parameters selected from the best F1 on validation set.

* NER (w.o): NER task without selecting hyper-parameters from the validation set.

| ﻿            Model            | NER.w        | NER.w.o 	| Analogy 	|
|------------------------------	|------------- | ------------------	|------------------	|
| BiLC3 + w2v                 	| 89.01     (5)  | 89.41         (6)   	|         0.4796 |
| BiLC3 + w2v_c2v             	| 89.46     (4)   | 89.46        (5)    	| 0.4796 |
| BiLC3 + fastText            	| 89.65     (3)   | 89.84        (4)    	|  0.4970 |
| BiLC3 + Elmo                	| 89.67 (2)  | **90.84**     (1)       	| 0.4999 |
| BiLC3 + Multi               	| **90.31**    (1)   | 90.83        (2)    	| 0.4907|
| BiLC3 + Bert_Base           	| 88.26     (6)  | 89.91         (3)    | 0.4609 |
| BiLC3 + Bert_Large         	| 88.05     (7)   | 88.58        (7)     | 0.4125|

# Embedding Extractor: To extract embedding vectors for other tasks.
- Input: (1) list of input embeddings, (2) a vocabulary file.
- Output: embedding vectors of the given vocab file in `.txt`, i.e., each line conains the embedding for a word. The file then be compressed in .gz format. This format is widely used in existing NLP Toolkits (e.g., Reimers et al. [1]).

## Extra options:
- `-input-c2v`: character embedding file
- `solveoov:1`: to solve OOV words of the 1st embedding. Similarly for more than one embedding: e.g., `solveoov:1:2`.


[1] Nils Reimers and Iryna Gurevych, Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging, 2017, http://arxiv.org/abs/1707.09861, arXiv.

## II. An application to Vietnamese
### 1. Multiple embedding models for Vietnamese NER-TASK are released in this work

| ﻿  Embedding Model           | Download Link (NER Task) |Download Link (General) | 
|------------------------------|---------------|---------------|
| w2v                          | [Link1]() | [Link1]() |
| w2v_c2v                      | [Link2]() | [Link2]() |
| fastText                     | [Link3]()| [Link3]() |
| Elmo                         | [Link4]() | [Link4]() |
| Multi                        | [Link5]() | [Link5]() |
| Bert_base                    | [Link6]() | [Link6]() |
| Bert_large                   | [Link7]()  | [Link7]() |

