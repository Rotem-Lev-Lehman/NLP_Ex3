from evaluator import Evaluator
from logistic_regression_classifier import LogisticRegressionClassifier
from svm_classifier import SVMClassifier
from xg_boost_classifier import XGBoostClassifier
from ff_nn_classifier import FFNNClassifier
from rnn_classifier import RNNClassifier
from data_manager import DataManager, fix_max_length

print('Starting')
best_hyper_parameters_solver = {}
best_score_solver = {}
classifiers = [LogisticRegressionClassifier(), SVMClassifier(), XGBoostClassifier(), FFNNClassifier(), RNNClassifier()]
for clf in classifiers:
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
    best_score = evaluator.optimize_hyper_parameters(X_train, y_train, cv=3, scoring='f1')
    print('Done')
    print(f'The classifier {clf.clf_name} gave us a best 3-fold score of: {best_score}')
    print(clf.hyper_parameters)
    best_hyper_parameters_solver[clf.clf_name] = clf.hyper_parameters
    best_score_solver[clf.clf_name] = best_score

print('**************************************************************************************************')
print('Done')
print('Results:')
for clf in classifiers:
    print(f'Current classifier: {clf.clf_name}')
    print(f'\tThe best F1 score is: {best_score_solver[clf.clf_name]}')
    print(f'\tThe best parameters are: {best_hyper_parameters_solver[clf.clf_name]}')
