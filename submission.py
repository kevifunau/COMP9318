from sklearn import svm
import numpy as np
import helper
from collections import defaultdict
from math import log


def findTop20(line,word_weight):
    ww = [(word, word_weight[word]) for word in line]
    ww = [x[0] for x in sorted(ww, key=lambda x: x[1], reverse=True)]
    r = []
    for word in ww:
        if word not in r:
            r.append(word)
        if len(r) == 20:
            break
    return r

def DeleteWordinLIne(line, deletelist):
    r = []
    for word in line:
        if word not in deletelist:
            r.append(word)
    assert len(set(line)) - len(set(r)) == 20
    return r


def fool_classifier(test_data): ## Please do not change the function defination...

    strategy_instance = helper.strategy()

    ######### feature extraction ###########
    vacub_set = list(set(words for line in strategy_instance.class0 for words in line).union(
        set(words for line in strategy_instance.class1 for words in line)))
    vacub_dict = {vacub_set[i]: i for i in range(len(vacub_set))}
    document_frequency = {vacub_set[i]: 0 for i in range(len(vacub_set))}
    row_length = {}
    v_k_dict = {v: k for k, v in vacub_dict.items()}

    # build feature_matrix for train_x
    feature_matrix = np.zeros((len(strategy_instance.class0) + len(strategy_instance.class1), len(vacub_dict)),
                              dtype=np.float64)
    # build class_matrix for train_y
    train_y = np.empty((len(strategy_instance.class0) + len(strategy_instance.class1)), dtype=np.int16)

    iter = 0
    while iter < len(strategy_instance.class0):
        for line in strategy_instance.class0:
            # set word value
            for word in line:
                if (vacub_dict.get(word) != None):
                    feature_matrix[iter, vacub_dict.get(word)] += 1
            for word in list(set(line)):
                document_frequency[word] += 1

            # set class value
            train_y[iter] = -1
            row_length[iter] = len(line)
            iter += 1

    while iter < len(strategy_instance.class0) + len(strategy_instance.class1):
        for line in strategy_instance.class1:
            # set word value
            for word in line:
                if (vacub_dict.get(word) != None):
                    feature_matrix[iter, vacub_dict.get(word)] += 1
            for word in list(set(line)):
                document_frequency[word] += 1
            # set class value
            train_y[iter] = 1
            row_length[iter] = len(line)
            iter += 1

    ####### TF-IDF ###########
    N = len(strategy_instance.class0) + len(strategy_instance.class1)

    row, col = feature_matrix.shape
    for i in range(row):
        for j in range(col):
            tf = feature_matrix[i, j]
            if tf > 0:
                word = v_k_dict[j]
                feature_matrix[i, j] = (tf / row_length[i]) * (log(1 + N / document_frequency[word], 2))


    ############# SVM TRAINING ###########


    # define params for SVM model
    parameters={}
    parameters['kernel'] = "linear"
    parameters['gamma'] = "auto"
    parameters['C'] = 1.0
    parameters['degree'] =3
    parameters['coef0'] = 0


    clf = strategy_instance.train_svm(parameters,feature_matrix,train_y)



    ########## generate weights of each words #############
    w_list = clf.coef_.transpose()
    w_list = [(w_list[i], i) for i in range(len(w_list))]
    word_weight = defaultdict(int)

    for k, v in w_list:
        word_weight[v_k_dict[v]] = k[0]

    weightorder = [x for x in sorted(word_weight.items(), key=lambda x: x[1])]


    #########  input text_data #############

    class_1_test = []
    test_data = './test_data.txt'

    with open(test_data, "r") as f:
        f = f.readlines()
        for line in f:
            class_1_test.append(line.strip().split(" "))



    ############# modified text ##################

    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    ## You can check that the modified text is within the modification limits.

    modified_data = './modified_data.txt'
    with open(modified_data, "w") as f:
        for line in class_1_test:
            deletelist = findTop20(line,word_weight)
            newline = DeleteWordinLIne(line, deletelist)
            assert len((set(line) - set(newline)) | (set(newline) - set(line))) == 20
            f.write(" ".join(newline) + "\n")


    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.




if __name__ == "__main__":
    test_data = './test_data.txt'
    strategy_instance = fool_classifier(test_data)

