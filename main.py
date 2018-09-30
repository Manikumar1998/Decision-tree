import sys
import time
import random
import numpy as np
from collections import Counter

class Node(object):
    def __init__(self, feature, value, left=None, right=None, _class=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self._class = _class

def split(dataset, f_index, value):
    left = {"X":[], "Y": []}
    right = {"X": [], "Y": []}
    X = dataset["X"]
    Y = dataset["Y"]
    
    for x,y in zip(X,Y):
        if x[f_index] <= value:
            left["X"].append(x)
            left["Y"].append(y)
            
        else:
            right["X"].append(x)
            right["Y"].append(y)
    groups = {'left':left, 'right':right}
    return groups

def get_entropy_of_split(dataset, f_index, value, classes):
    groups_dict = split(dataset, f_index, value)
    groups = []
    groups.append(groups_dict['left'])
    groups.append(groups_dict['right'])

    entropy = 0
    total_size = len(dataset['Y'])
    
    for group in groups:
        class_count = Counter(group['Y'])
        length_group = float(len(group["X"]))
        normal_size = float(length_group)/total_size
        group_entropy = 0
    
        for _class in classes:
            if not class_count.has_key(_class):
                continue
            count = class_count[_class]
            prob = float(count)/length_group
            group_entropy -= prob*np.log2(prob)
        entropy += normal_size*group_entropy
    return entropy, groups_dict

def cal_gain(dataset, classes, f_index, value):
    classes_count = Counter(dataset['Y'])
    total_count = len(dataset['Y'])
    initial_entropy = 0

    for _class in classes:
        prob = float(classes_count[_class])/total_count
        initial_entropy -= prob*np.log2(prob)

    entropy_of_split, groups = get_entropy_of_split(dataset, f_index, value, classes)
    gain = initial_entropy-entropy_of_split
    return gain, groups


def majority_voting(dataset):
    Y = dataset['Y']
    Y_count = Counter(Y)
    _class = max(Y_count)
    return _class
    

def construct_decision_tree(dataset, limits, classes, features):
    #Stop when all belong to same class
    if len(classes) == 1:
        return Node(None,
                    None,
                    _class=classes[0])

    #Stop when no features are left
    if not features:
        _class = majority_voting(dataset)
        return Node(None,
                    None,
                    _class=_class)

    #Stop when no samples left
    if not dataset['X']:
        return None
        
    X = dataset['X']
    Y = dataset['Y']

    _gains = []
    for f_index in features:
        information_gain = []
        _min = limits[f_index]['min']
        _max = limits[f_index]['max']
        for value in np.linspace(_min, _max, 100):
            gain, groups = cal_gain(dataset, classes, f_index, value)
            information_gain.append({'value':value,
                                     'gain': gain,
                                     'groups':groups})
            
        max_gain_pair = max(information_gain, key=lambda i:i['gain'])
        _gains.append({'f_index':f_index,
                       'max_gain':max_gain_pair['gain'],
                       'value':max_gain_pair['value'],
                       'groups':max_gain_pair['groups']})
    
    selected = max(_gains, key=lambda i:i['max_gain'])

    node = Node(selected['f_index'],
                selected['value'],
                _class=None)

    new_features = []
    for f_index in features:
        if f_index != selected['f_index']:
            new_features.append(f_index)

    left_data = selected['groups']['left']
    left_limits = get_limits(left_data, new_features)
    left_classes = get_classes(left_data)
    
    right_data = selected['groups']['right']
    right_limits = get_limits(right_data, new_features)
    right_classes = get_classes(right_data)
    
    node.left =  construct_decision_tree(left_data,
                                         left_limits,
                                         left_classes,
                                         new_features)

    node.right = construct_decision_tree(right_data,
                                         right_limits,
                                         right_classes,
                                         new_features)
    return node

def get_limits(dataset, features):
    X = dataset['X']
    limits = {}
    for index in features:
        limits[index] = {'min':None, 'max':None}

    for x in X:
        for index in features:
            if limits[index]['min'] == None:
                limits[index]['min'] = x[index]
            elif x[index] < limits[index]['min']:
                limits[index]['min'] = x[index]
                
            if limits[index]['max'] == None:
                limits[index]['max'] = x[index]
            elif x[index] > limits[index]['max']:
                limits[index]['max'] = x[index]
    return limits

def get_classes(dataset):
    Y = dataset['Y']
    return list(set(Y))

def classify(root, X_data):
    classified = []
    for x in X_data:
        node = root
        while(node != None):
            if node._class != None:
                classified.append(node._class)
                break
            else:
                if x[node.feature] <= node.value:
                    node = node.left
                else:
                    node = node.right
    return classified

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
        vector = map(str.strip, vector.split())
        try:
            Y_train.append(int(vector.pop(-1)))
            X_train.append(map(float, vector))
        except:
            print vector

    for vector in data:
        vector = map(str.strip, vector.split())
        Y_test.append(int(vector.pop(-1)))
        X_test.append(map(float, vector))

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    with open("sensorless_drive_data.txt", 'r') as fp:
    #with open("banknote_auth_data.txt", 'r') as fp:
    #with open("test_data.txt", 'r') as fp:
        data = fp.readlines()
    m = len(data)
    X_train, Y_train, X_test, Y_test = extract_data(data, 90)

    #REMOVE for actual data
    X_train.extend(X_test)
    Y_train.extend(Y_test)
    
    features = range(len(X_train[0]))
    dataset = {'X':X_train, 'Y':Y_train}
    classes = get_classes({'X':X_train+X_test, 'Y':Y_train+Y_test})
    
    #Should we get min of only X_train or X_train+X_test?
    limits = get_limits(dataset, features)
    root = construct_decision_tree(dataset, limits, classes, features)
    # print root.feature, root.value, root._class
    # print root.right.feature, root.right.value, root.right._class
    # print root.left.feature, root.left.value, root.left._class
    # print root.left.left.feature, root.left.left.value, root.left.left._class
    # print root.left.right.feature, root.left.right.value, root.left.right._class
    # print classify(root, [[8, 290, 38],
    #                       [6, 200, 45],
    #                       [8, 160, 41],
    #                       [4, 20, 1],
    #                       [6, 78, 8],
    #                       [0, 250, 36]])
    Y_ = classify(root, X_test)
    count = 0
    for i, j in zip(Y_, Y_test):
        if i == j:
            count += 1
    print count/float(len(Y_test))
