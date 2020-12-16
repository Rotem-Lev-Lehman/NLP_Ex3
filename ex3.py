from evaluator import Evaluator
from logistic_regression_classifier import LogisticRegressionClassifier
from svm_classifier import SVMClassifier
from xg_boost_classifier import XGBoostClassifier
from ff_nn_classifier import FFNNClassifier
from rnn_classifier import RNNClassifier
from data_manager import DataManager, fix_max_length
from time import time

print('Starting')
start_time = time()
clf = FFNNClassifier()
print(f'Initialized classifier {clf.clf_name}')
evaluator = Evaluator(clf)
print('Initialized evaluator')
dm_train = DataManager('trump_train.tsv', is_train=True, algorithm_name=clf.clf_name)
dm_test = DataManager('trump_test.tsv', is_train=False, algorithm_name=clf.clf_name)

print('Initialized Data manager')
dm_train.run_first_preprocessing_flow()
dm_test.run_first_preprocessing_flow()
fix_max_length(dm_train, dm_test)
X_train, y_train = dm_train.complete_preprocessing_flow()
X_test, _ = dm_test.complete_preprocessing_flow()
if clf.clf_name == 'RNN':
    clf.sequence_length = dm_train.max_length
print('Cleaned X, y')
best_score = evaluator.optimize_hyper_parameters(X_train, y_train, cv=3, scoring='accuracy')
end_time = time()
print('Done')
print(f'It took a total of {end_time - start_time} seconds')
print(f'The classifier {clf.clf_name} gave us a best 10-fold score of: {best_score}')
print(clf.hyper_parameters)

# Logistic Regression =     0.825
# SVM =                     0.857
# XGB =                     0.864
# FF-NN =                   0.837

# TODO - Make FF-NN and RNN (or just America) great again!
# TODO - Look at the assignment's requirements.
