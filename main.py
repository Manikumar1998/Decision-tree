import sys
import time
import random
import numpy as np

class Node(object):
    def __init__(self, feature, value, left, right, _class=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self._class = _class

def split( dataset, feature, value):
    Left = { "X": [], "Y": [] }
    Right = { "X": [], "Y": [] }
    a = dataset["Left"]
    b = dataset["Right"]
    for i in range(len(a)):
        if (a[i][feature] > value):
            Left["X"].append(a[i][feature])
            Left["Y"].append(b[i])
            
        else:
            Right["X"].append(a[i][feature])
            Right["Y"].append(b[i])
            
    return Left, Right

# considering groups as a list of left and right
def get_entropy_of_split (left,right,classes):

    entropy = 0
    groups = [left,right]
    total_size = len(left["X"]) + len(right["Y"])
        
    for i in groups:
        length_group = len(i["X"])
        normal_size = length_group / total_size
        group_sum = 0
        for j in classes:
            count = 0
            for k in i["Y"]:
                if (k== j):
                    count += 1
            group_sum -= count/length_group*math.log(cpunt/length_group,2)
        entropy += normal_size*group_sum
    return entropy

def cal_entropy(a, b, c, d):
    #TO-DO: calculate the entropy
    return None
        
def construct_decision_tree(dataset, limits, classes, m, n):
    X = dataset['X']
    Y = dataset['Y']

    for f_index in xrange(n):
        _min = limits[f_index]['min']
        _max = limits[f_index]['max']
        for value in xrange(np.linspace(_min, _max, 100)):
            entropy = cal_entropy(dataset, classes, f_index, value)

def get_limits(X, n):
    limits = []
    for _ in xrang(n):
        limits.append({'min':None, 'max':None})

    for vector in X:
        for index, value in enumerate(vector):
            if not limits[index]['min'] or (value < limits[index]['min']):
                limits[index]['min'] = value
                
            if not limits[index]['max'] or (value > limits[index]['max']):
                limits[index]['max'] = value

    return limits
    
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

    classes = list(set(Y_train+Y_test))
    return X_train, Y_train, X_test, Y_test, classes

if __name__ == "__main__":
    with open("banknote_auth_data.txt", 'r') as fp:
        data = fp.readlines()
    m = len(data)
    X_train, Y_train, X_test, Y_test, classes = extract_data(data, 90)
    n = len(X_train[0])
    dataset {'X':X_train, 'Y':Y_train}

    #Should we get min of only X_train or X_train+X_test?
    limits = get_limits(X_train, n)
    construct_decision_tree(dataset, limits, classes, m, n)

    
