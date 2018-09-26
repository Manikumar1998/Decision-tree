import sys
import time
import random

class Node(object):
    def __init__(self, feature, value, left, right, _class=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self._class = _class

def extract_data(data, ratio):
    split_len = len(data)*ratio/100
    training_set = []
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    
    while(len(training_set) < split_len):
        index = random.randint(0, len(data)-1)
        training_set.append(data.pop(index))

    for vector in training_set:
        vector = map(str.strip, vector.split(','))
        Y_train.append(int(vector.pop(-1)))
        X_train.append(map(float, vector))

    for vector in data:
        vector = map(str.strip, vector.split(','))
        Y_test.append(int(vector.pop(-1)))
        X_test.append(map(float, vector))

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    with open("banknote_auth_data.txt", 'r') as fp:
        data = fp.readlines()
    m = len(data)
    X_train, Y_train, X_test, Y_test = extract_data(data, 90)
    n = len(X_train[0])

    

    
