from evaluator import Evaluator
from logistic_regression_classifier import LogisticRegressionClassifier
from svm_classifier import SVMClassifier
from xg_boost_classifier import XGBoostClassifier
from ff_nn_classifier import FFNNClassifier
from data_manager import DataManager
from time import time

print('Starting')
start_time = time()
clf = FFNNClassifier()
print('Initialized LR')
evaluator = Evaluator(clf)
print('Initialized evaluator')
dm = DataManager('trump_train.tsv', is_train=True, algorithm_name=clf.clf_name)
print('Initialized Data manager')
X, y = dm.run_preprocessing_flow()
print('Cleaned X, y')
best_score = evaluator.optimize_hyper_parameters(X, y)
end_time = time()
print('Done')
print(f'It took a total of {end_time - start_time} seconds')
print(f'The classifier {clf.clf_name} gave us a best 10-fold score of: {best_score}')
print(clf.hyper_parameters)

# Logistic Regression =     0.825
# SVM =                     0.857
# XGB =                     0.864
# FF-NN =                   0.837

# TODO - Add FF-NN and RNN.
# TODO - Look at the assignment's requirements.
