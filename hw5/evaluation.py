'''
CAPP30254 HW5
Xuan Bu
Evaluation
'''

from sklearn.metrics import *
import numpy as np
import matplotlib.pyplot as plt


def generate_ROC_graph(pred_scores, test_target, model):
    '''
    Plot the graph of Receiver Operating Characteristic.
    Inputs:
        pred_scores: (array) the 
        test_target: (dataframe) testing data of target variable
        model: (str) name of the classifier
    '''
    fpr, tpr, threshold = roc_curve(test_target, pred_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label = 'ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of ' + str(model))
    plt.legend(loc = 'lower right')
    plt.savefig('ROC Graph of ' + str(model))
    plt.show()
    plt.close()


def accuracy_at_k(y_true, y_scores, k):
    '''
    Get the accuracy score at different percentage of porjects
    predicted true.
    Inputs:
        y_true: (dataframe) testing data of target variable
        y_scores: (array) the probabilities of being predicted true
        k: (int) percentage of projects predicted true
    Returns:
        (float) accuracy score
    '''
    y_scores_sorted, y_true_sorted = \
            joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)

    return accuracy_score(y_true_sorted, preds_at_k)


def roc_auc_at_k(y_true, y_scores, k):
    '''
    Get the roc-auc score at different percentage of porjects
    predicted true.
    Inputs:
        y_true: (dataframe) testing data of target variable
        y_scores: (array) the probabilities of being predicted true
        k: (int) percentage of projects predicted true
    Returns:
        (float) roc-auc score
    '''
    y_scores_sorted, y_true_sorted = \
            joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)

    return roc_auc_score(y_true_sorted, preds_at_k)


###########################################################################
### The following functions are from Professor Rayid Ghani's Github Repo. #
### Source: https://github.com/rayidghani/magicloops                      #
###########################################################################

def generate_precision_recall_curve(pred_scores, test_target, model):
    '''
    Plot the graph of the tradeoff between precision and recall under
    different thresholds.
    Inputs:
        pred_scores: (array) the probabilities of being predicted true
        test_target: (dataframe) testing data of target variable
        model: (str) name of the classifier
    '''
    precision, recall, thresholds = \
                precision_recall_curve(test_target, pred_scores) 
    precision = precision[:-1]
    recall = recall[:-1]
    pct_above_per_thresh = []
    for t in thresholds:
        num_above_thresh = len(pred_scores[pred_scores >= t])
        pct_above_thresh = num_above_thresh / float(len(pred_scores))
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    # plot
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision, 'b')
    ax1.set_xlabel('Percent of Population')
    ax1.set_ylabel('Precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall, 'r')
    ax2.set_ylabel('Recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    plt.title('The Tradeoff between Precision and Recall of '+str(model))
    plt.savefig('The Tradeoff between Precision and Recall of '+str(model))
    plt.show()
    plt.close()


def precision_at_k(y_true, y_scores, k):
    '''
    Get the precision score at different percentage of porjects
    predicted true.
    Inputs:
        y_true: (dataframe) testing data of target variable
        y_scores: (array) the probabilities of being predicted true
        k: (int) percentage of projects predicted true
    Returns:
        (float) precision score
    '''
    y_scores_sorted, y_true_sorted = \
            joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)

    return precision_score(y_true_sorted, preds_at_k)


def recall_at_k(y_true, y_scores, k):
    '''
    Get the recall score at different percentage of porjects
    predicted true.
    Inputs:
        y_true: (dataframe) testing data of target variable
        y_scores: (array) the probabilities of being predicted true
        k: (int) percentage of projects predicted true
    Returns:
        (float) recall score
    '''
    y_scores_sorted, y_true_sorted = \
            joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)

    return recall_score(y_true_sorted, preds_at_k)


def generate_binary_at_k(y_scores, k):
    '''
    Get the labels of 1 or 0 at different percentage of porjects
    predicted true.
    Inputs:
        y_scores: (array) the probabilities of being predicted true
        k: (int) percentage of projects predicted true
    Returns:
        (list) of projects being predicted true or false
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = \
            [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary


def joint_sort_descending(l1, l2):
    '''
    Given two arrays, then sort them in descending order.
    Inputs:
        l1: first array
        l2: second array
    Returns:
        sorted arrays
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

