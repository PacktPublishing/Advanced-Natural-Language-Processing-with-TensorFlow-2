# Installation and Setup Instructions for Code

Instructions for setting an environment for the code in the book are below. These instructions :

- Have been tested on MacOS 10.15 and Ubuntu 18.04.3 LTS. You may have to translate these instructions for WIndows specific vagaries
- Only cover CPU version of TensorFlow. For latest GPU install instructions, please follow <https://www.tensorflow.org/install/gpu>

Installation uses Anaconda distribution and `pip`. It is assumed that Anaconda is setup and ready to go on your machine. Note that we use some new and some uncommon packages. These packages may not be available through conda. We will use `pip` in such cases.

Notes:

- On MacOS: conda 4.6.8, pip 20.1.1
- On Ubuntu: conda 4.6.11, pip 20.0.2

**Step 1**: Create a new conda environment with Python 3.7.5

```
$ conda create -n  python==3.7.5
```

The environment is named `tf23nlp` but feel free to use your own name and make sure you use that in the following steps. I like to prefix my environment names with the version of TensorFlow being used and I suffix a 'g' if that environment has GPU version of the library. As you can probably infer, we are going to use TensorFlow 2.1

**Step 2**: Activate the environment and install the following packages

```
$ conda activate tf23nlp
$  conda install pandas==1.0.1 numpy==1.18.1
```

This installs Numpy and Pandas the libraries in newly created environment.

**Step 3**: Install TensorFlow 2.1 . To do this, we will need to use pip. As of writing, conda was still at 2.0\. TF has been moving quite fast. In general, conda distributions a little behind the latest versions.

```
(tf23nlp) $ pip install tensorflow==2.3
```

Please note that these instructions are for the CPU version of TensorFlow. For GPU installation instructions, please refer to <https://www.tensorflow.org/install/gpu> .

**Step 4**: Install Jupyter notebook - feel free to install the latest version:

```
(tf23nlp) $ conda install Jupyter
```

Rest of the installation instructions are about specific libraries used in specific chapters. If you have trouble installing through Jupyter Notebook, you can install from command line.

## Chapter 1 Installation Instructions

No specific instructions as code for this chapter run on [Google Colab](colab.research.google.com)

## Chapter 2 Installation Instructions

`tfds` package needs to be installed:

```
(tf23nlp) $ pip install tensorflow_datasets
```

We will `tfds` in most of the chapters going forward.

## Chapter 3 Installation Instructions

1. Install `matplotlib` via:

  ```
  (tf23nlp) $ conda install matplotlib==3.1.3
  ```

  A newer version may work as well.

2. Install TensorFlow Addons package for Viterbi decoding

  ```
  (tf23nlp) $ pip install tensorflow_addons==0.11.2
  ```

  Note that this package is not available through conda.

## Chapter 4 Installation Instructions

This chapter requires installation of `sklearn`:

```
(tf23nlp) $ conda install scikit-learn==0.23.1
```

HuggingFace's Transformers library needs to be installed as well.

```
(tf23nlp) $ pip install transformers==3.0.2
```

## Chapter 5 Installation Instructions

None required

## Chapter 6 Installation Instructions

None required.

## Chapter 7 Installation Instructions

Visual QA

Processing images need the Pillow library, which is the friendly version of Python Imaging Library. It can be installed like so:

```
(tf23nlp) conda install pillow==7.2.0
```

TQDM is a nice utility to display progress bars while executing long loops:

```
(tf23nlp) $ conda install tqdm==4.47.0
```

## Chapter 8 Installation Instructions

Snorkel needs to be installed. As of writing, the version of Snorkel installed was 0.9.5\. Note that this version of Snorkel uses older versions of Pandas and TensorBoard. You should be able to safely ignore these issues. If you continue to face conflicts in your environment, then I suggest creating a separate Snorkel specific environment. Run the labeling functions in that environment and store the outputs as a separate csv file. TensorFlow training can be run by switching back to the `tf23nlp` environment and loading the labelled data in.

```
(tf23nlp) $ pip install snorkel==0.9.5
```

BeautifulSoup for parsing HTML tags out of the text.

```
(tf23nlp) $ conda install beautifulsoup4==4.9
```

There is an optional section that plots Word clouds. This needs the following package to be installed:

```
(tf23nlp) $ pip install wordcloud==1.8
```

Note that this chapter also uses NLTK, which was used in the first chapter.

## Chapter 9 Installation Instructions

None.
