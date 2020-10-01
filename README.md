# Advanced Natural Language Processing with TensorFlow 2

2019 has been a watershed moment for NLP with transformer and attention-based networks. This is as transformational for NLP as AlexNet was for computer vision in 2012\. Tremendous advances in Natural Language processing have been made in the last couple of years, and we are now moving from research labs into applications. These advances span the domains of Natural Language Understanding (NLU), Natural Language Generation (NLG) and Natural Language Interaction (NLI). With so much research in all of these domains, it can be a daunting task to understand the exciting developments in various domains inside NLP.

This book is focused on cutting edge applications in the fields of natural language processing, language generation, and dialogue systems. It provides the concepts of pre-processing text using techniques such as tokenization, parts of speech tagging, lemmatization, Named Entity Recognition (NER) using popular libraries such as Stanford NLP and SpaCy. Taking a very practical, application-focussed perspective, the book covers key emerging areas such as generating text for use in sentence completion and text summarization, bridging images and text by generating captions for images and answering common sense questions for them, and managing dialogue aspects of chatbots. It covers one of the most important reasons behind recent advances of NLP - transfer learning and fine tuning. Unlabeled textual data is easily available, however labelling this data is costly. This book covers practical techniques that can simplify labeling of textual data.

By the end of the book, the reader will have an advanced knowledge of the tools, techniques and deep learning architectures used to solve complex NLP problems. The book will cover encoder-decoder networks, LSTMs and BiLSTMs, CRFs, BERT, Transformers and other key technology pieces using TensorFlow. They will have working code that can be adapted to their own use cases. We hope that the readers will be able to even do novel state-of-the-art research using the skills gained.

The book uses TensorFlow 2.3 and Keras extensively. Several advanced TensorFlow techniques are also covered such as:

- Custom learning rate schedules
- Custom loss functions
- Custom layers
- Custom training loops
- Subword encoding for embeddings
- `tensorflow_datasets` package for downloading and managing datasets
- `tf.data.Dataset` usage and performance optimization
- Model checkpointing

The book is organized in the following chapters (with links to code):

1. [Essentials of NLP](chapter1-nlp-essentials):
2. [Understanding Sentiment in Natural Language with BiLSTMs](chapter2-nlu-sentiment-analysis-bilstm)
3. [Named Entity Recognition (NER) with BiLSTMs, CRFs and Viterbi Decoding](chapter3-ner-with-lstm-crf)
4. [Transfer Learning with BERT](chapter4-Xfer-learning-BERT)
5. [Generating Text With RNNs and GPT-2](chapter5-nlg-with-transformer-gpt)
6. [Text Summarization with seq2seq Attention and Transformer Networks](chapter6-textsum-seq2seq-attention-transformer)
7. [Multi-modal Networks and Im-age Captioning with ResNets and Transformer](chapter-7-image-cap-multimodal-transformers)
8. [Weakly Supervised Learning for Classification with Snorkel](chapter-8-weak-supervision-snorkel)
9. [Building Conversational AI Applications with Deep Learning](chapter-9-conversational-agents)

Seminal papers relevant to the chapter or referenced in the chapter can also be found in the appropriate chapter directory.

## Installation

[install.md](install,md) contains all the installation instructions for running the code. The book has been written and tested using Ubuntu 18.04 LTS, with a NVIDIA RTX 3070 GPU running Python 3.7 and TensorFlow 2.3, which was the latest version as of writing.

Author: [Ashish Bansal](linkedin.com/in/bansalashish)
