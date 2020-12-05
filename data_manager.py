import pandas as pd
import numpy as np
from utils import *
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from embedding_manager import embedding_dict


class DataManager:

    def __init__(self, path, is_train=True, use_stemming=False, remove_stopwords=True, algorithm_name=None):
        """ Initializes a DataManager object.

        :param path: the path to the data file we want to analyze
        :type path: str
        :param is_train: indicates whether we are using train data(True) or test data(False). Defaults to True
        :type is_train: bool
        :param use_stemming: indicates whether we are using stemming on the text(True) or not(False). Defaults to False
        :type use_stemming: bool
        :param remove_stopwords: indicates whether we are removing stopwords from the text(True) or not(False).
                                    Defaults to True
        :type remove_stopwords: bool
        :param algorithm_name: the algorithm name we will be using this data with.
        :type algorithm_name: str
        """
        self.path = path
        self.is_train = is_train
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.algorithm_name = algorithm_name
        self.df = None
        self.y = None
        self.X = None

    def run_preprocessing_flow(self):
        """ Runs the entire preprocessing flow.
         And returns the preprocessed DataFrames (X, y) ready to be used in the training section.

        :return: the preprocessed DataFrames (X, y) ready to be used in the train/test process
        :rtype: tuple
        """
        self.read_df()
        self.clean_data_and_extract_features()
        self.handle_embedding_features()
        self.set_target()
        return self.X, self.y

    def clean_data_and_extract_features(self):
        """ Cleans the data given, and extract relevant features from it.
        """
        self.time_stamp_to_features()
        self.extract_text_features()
        self.create_dummy_features()
        self.clean_tweet_text()

    def read_df(self):
        """ Reads the df from the given file path, with the preferences of train/test.
        """
        if self.is_train:
            self.df = pd.read_csv(self.path, header=None, sep='\t',
                             names=['tweet id', 'user handle', 'tweet text', 'time stamp', 'device'])
            self.df.drop(labels=['tweet id'], axis=1, inplace=True)
        else:
            self.df = pd.read_csv(self.path, header=None, sep='\t',
                             names=['user handle', 'tweet text', 'time stamp'])

    def time_stamp_to_features(self):
        """ Extracts time features from the time stamp column.
        """
        self.df['time stamp'] = pd.to_datetime(self.df['time stamp'])
        self.df["year"] = self.df["time stamp"].dt.year
        self.df["month"] = self.df["time stamp"].dt.month
        self.df["day"] = self.df["time stamp"].dt.day
        self.df["dayOfWeek"] = self.df["time stamp"].dt.dayofweek
        self.df["dayOfYear"] = self.df["time stamp"].dt.dayofyear
        self.df["hour"] = self.df["time stamp"].dt.hour
        self.df["minute"] = self.df["time stamp"].dt.minute
        self.df.drop(['time stamp'], axis=1, inplace=True)

    def extract_text_features(self):
        """ Extract features from the tweet text column.
        """
        self.df['contains_URL'] = self.df['tweet text'].apply(lambda text: contains_URL(text))
        self.df['caps_locked_count'] = self.df['tweet text'].apply(lambda text: count_caps_locked_words(text))
        self.df['count_hashtags'] = self.df['tweet text'].apply(lambda text: count_hashtags(text))
        self.df['count_mentions'] = self.df['tweet text'].apply(lambda text: count_mentions(text))

    def create_dummy_features(self):
        """ Generates dummy variables for the user-handle column.
        """
        dummies = pd.get_dummies(self.df['user handle'])
        self.df = pd.concat([self.df, dummies], axis=1)
        self.df.drop(labels=['user handle'], axis=1, inplace=True)

    def clean_tweet_text(self):
        """ Cleans the text in the tweet-text column.
        """
        punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        stop_words = stopwords.words('english')
        porter = PorterStemmer()

        self.df['tweet text'] = self.df['tweet text'].apply(lambda tweet: tweet.lower())
        self.df['tweet text'] = self.df['tweet text'].apply(
            lambda tweet: "".join([char for char in tweet if char not in punctuation]))
        self.df['tweet text'] = self.df['tweet text'].apply(lambda tweet: word_tokenize(tweet))
        if self.remove_stopwords:
            self.df['tweet text'] = self.df['tweet text'].apply(
                lambda words: [word for word in words if word not in stop_words])
        if self.use_stemming:
            self.df['tweet text'] = self.df['tweet text'].apply(
                lambda filtered_words: [porter.stem(word) for word in filtered_words])

    def set_target(self):
        """ Creates the self.X and self.y DataFrames.
        """
        self.y = (self.df['device'] != 'android').astype(int)  # 0=Trump, 1=Not Trump
        self.X = self.df.iloc[:, self.df.columns != 'device']

    def handle_embedding_features(self):
        """ Handles the embedding features from the tweet-text column.
            The embedding features that we will create depends on the self.algorithm_name in use.
        """
        if self.algorithm_name in ['SVM', 'Logistic Regression', 'XGBoost']:
            self.add_words_mean_embedding_features()
        else:
            # These algorithms are the FF-NN and RNN networks.
            raise Exception('Need to implement')

    def add_words_mean_embedding_features(self):
        """ Adds the embedding features of the tweet-text column to be the mean of all of it's words' embeddings.
        """
        self.df['tweet embedding'] = self.df['tweet text'].apply(
            lambda words: np.apply_along_axis(np.mean, 0, np.array(
                [embedding_dict[word] for word in words if word in embedding_dict.keys()])))

        # TODO - Need to think of something smarter than just to drop these columns:
        self.df = self.df.dropna()

        for i in range(len(self.df['tweet embedding'][0])):
            self.df[f'embedding_{i}'] = self.df['tweet embedding'].apply(lambda vector: vector[i])

        self.df.drop(labels=['tweet text', 'tweet embedding'], axis=1, inplace=True)

