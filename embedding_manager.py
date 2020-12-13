import csv
import numpy as np
import pandas as pd
import gzip
import shutil
import torch
# from gensim.models import Word2Vec


def zip_file(path_from, path_to):
    """ zips the file with a gz ending.

    :param path_from: the file to zip
    :type path_from: str
    :param path_to: the output filepath
    :type path_to: str
    """
    with open(path_from, 'rb') as f_in:
        with gzip.open(path_to, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def load_embedded_file(embedded_file_path):
    """ loads a zipped embedded file and creates the embedding_dict from it.

    :param embedded_file_path: the zipped embedded file path
    :type embedded_file_path: str
    :return: the embedding dict
    :rtype: dict
    """
    print('Creating embedding dictionary')
    embedding_dict = {}
    with gzip.open(embedded_file_path, mode="rt") as file1:
        all_lines = file1.readlines()
        for line in all_lines:
            split_line = line.strip().split()
            embedding_dict[split_line[0]] = np.array(split_line[1:]).astype(np.float32)
    print('Done creating embedding dictionary')
    return embedding_dict


def read_df(filepath, is_train):
    """ Reads the df from the given file path, with the preferences of train/test.

    :return: the tweet text column of the data
    :rtype: pd.Series
    """
    if is_train:
        df = pd.read_csv(filepath, header=None, sep='\t',
                         names=['tweet id', 'user handle', 'tweet text', 'time stamp', 'device'],
                         engine='python', quoting=csv.QUOTE_NONE)
        df.drop(labels=['tweet id'], axis=1, inplace=True)
    else:
        df = pd.read_csv(filepath, header=None, sep='\t',
                         names=['user handle', 'tweet text', 'time stamp'],
                         engine='python', quoting=csv.QUOTE_NONE)
    return df['tweet text']


def preprocess(tweet_text_series, use_stemming=False, remove_stopwords=True):
    """ Preprocesses the tweet text column.

    :param tweet_text_series: the series we want to preprocess
    :type tweet_text_series: pd.Series
    :param use_stemming: indicates whether we are using stemming on the text(True) or not(False). Defaults to False
    :type use_stemming: bool
    :param remove_stopwords: indicates whether we are removing stopwords from the text(True) or not(False).
                                Defaults to True
    :type remove_stopwords: bool
    :return: the preprocessed series
    :rtype: pd.Series
    """
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk import word_tokenize
    
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    stop_words = stopwords.words('english')
    porter = PorterStemmer()

    tweet_text_series = tweet_text_series.apply(lambda tweet: tweet.lower())
    tweet_text_series = tweet_text_series.apply(
        lambda tweet: "".join([char for char in tweet if char not in punctuation]))
    tweet_text_series = tweet_text_series.apply(lambda tweet: word_tokenize(tweet))
    if remove_stopwords:
        tweet_text_series = tweet_text_series.apply(
            lambda words: [word for word in words if word not in stop_words])
    if use_stemming:
        tweet_text_series = tweet_text_series.apply(
            lambda filtered_words: [porter.stem(word) for word in filtered_words])
    return tweet_text_series


def parse_legal_words(files):
    """ parses the legal words from the given files.

    :param files: a dictionary of: {'train': <train_filepath>, 'test': <test_filepath>}
    :type files: dict
    :return: a set of the legal words in the files
    :rtype: set
    """
    df_train = read_df(files['train'], is_train=True)
    df_test = read_df(files['test'], is_train=False)
    all_tweet_texts = pd.concat([df_train, df_test])
    all_tweet_texts = preprocess(all_tweet_texts)
    legal_words = set()
    for words_array in all_tweet_texts.values:
        for word in words_array:
            legal_words.add(word)
    return legal_words


def create_embedded_file(big_embedded_filepath, files, output_file):
    """ creates a zipped embedded file containing only the relevant words from the given tweets.

    :param big_embedded_filepath: a file that contains the pretrained embedded file
    :type big_embedded_filepath: str
    :param files: a dictionary of: {'train': <train_filepath>, 'test': <test_filepath>}
    :type files: dict
    :param output_file: the path to save the output file to
    :type output_file: str
    """
    legal_words = parse_legal_words(files)
    legal_lines = []
    with open(big_embedded_filepath, encoding='utf-8') as file1:
        lines = file1.readlines()
        for l in lines:
            split_line = l.rstrip().split()
            word = split_line[0]
            if word in legal_words:
                legal_lines.append(l)

    with open(output_file, 'w') as write_file:
        write_file.writelines(legal_lines)

    path_from = output_file
    path_to = f'{output_file}.gz'
    zip_file(path_from, path_to)

'''
def create_embedding_file_with_gensim():
    # define training data
    sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
                 ['this', 'is', 'the', 'second', 'sentence'],
                 ['yet', 'another', 'sentence'],
                 ['one', 'more', 'sentence'],
                 ['and', 'the', 'final', 'sentence']]
    # train model
    model = Word2Vec(sentences, min_count=1)
    # fit a 2d PCA model to the vectors
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()
'''

# create_embedded_file('glove.6B.300d.txt', files={'train':'trump_train.tsv',
#                                                  'test':'trump_test.tsv'}, output_file='embedded_file.txt')

embedding_dict = load_embedded_file('embedded_file.txt.gz')
PADDING_WORD = '<PAD>'
embedding_dict[PADDING_WORD] = np.array([0] * 300)
vocab = set(embedding_dict.keys())

word_to_ix = {}
weights = []
for i, word in enumerate(vocab):
    word_to_ix[word] = i
    weights.append(embedding_dict[word])

pretrained_weights = torch.Tensor(weights)
