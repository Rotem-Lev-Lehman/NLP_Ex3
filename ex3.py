from data_manager import DataManager
from xg_boost_classifier import XGBoostClassifier
from base_classifier import BaseClassifier
import pickle


"""
This is the driver file as needed in the assignment.
"""


def load_best_model():
    """ loads the best saved model (The trained XGBoost model file it's best hyper-parameters).

    :return: our best performing model that was saved as part of the submission bundle
    :rtype: XGBoostClassifier
    """
    with open('best_model.pickle', 'rb') as f:
        clf = pickle.load(f)
    return clf


def train_best_model():
    """ training a XGBoost classifier from scratch with it's best hyper-parameters.

    :return: a trained XGBoost classifier built with the best performing hyper-parameters.
    :rtype: XGBoostClassifier
    """
    clf = XGBoostClassifier()
    clf.set_best_hyper_parameters()  # sets the best hyper-parameters that were found in the optimization stage.
    dm_train = DataManager('trump_train.tsv', is_train=True, algorithm_name=clf.clf_name)
    # we had two stages of the preprocessing flow because of the padding of the text features used in the NN algorithms:
    dm_train.run_first_preprocessing_flow()
    X_train, y_train = dm_train.complete_preprocessing_flow()
    # fit the classifier using all of the training data:
    clf.fit(X_train, y_train)
    return clf


def predict(m, fn):
    """ returns a list of 0s and 1s, corresponding to the lines in the specified file.

    :param m: the trained model
    :type m: BaseClassifier
    :param fn: the full path to a file in the same format as the test set
    :type fn: str
    :return: a list of 0s and 1s, corresponding to the lines in the specified file
    :rtype: list
    """
    dm_test = DataManager(fn, is_train=False, algorithm_name=m.clf_name)
    dm_test.run_first_preprocessing_flow()
    X_test, _ = dm_test.complete_preprocessing_flow()
    return m.predict(X_test)


def save_best_model(clf):
    """ saves the best model to a pickle file to be able to load later.

    :param clf: the classifier to save
    :type clf: XGBoostClassifier
    """
    with open('best_model.pickle', 'wb') as f:
        pickle.dump(clf, f)


def save_results(results):
    """ saves the results list to a file.

    :param results: the results list to save
    :type results: list
    """
    with open('208965814_311272264.txt', 'w') as f:
        f.write(' '.join(str(x) for x in results))

# In order to train the best classifier and save it's results and pickle file, run the following code:
'''
clf = train_best_model()
results = predict(m=clf, fn='trump_test.tsv')
save_results(results)
save_best_model(clf)
'''
# In order to load the best classifier and get it's predictions on the test set, run the following code:
'''
clf = load_best_model()
prediction = predict(m=clf, fn='trump_test.tsv')
print(' '.join(str(x) for x in prediction))
'''
