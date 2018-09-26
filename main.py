import sys

if __name__ == "__main__":
    fp = open("banknote_auth_data.txt", 'r')
    data = fp.readlines()
    m = len(data)
    Y = []
    X = []
    print len(data)
    for vector in data:
        print map(str.strip, vector.split(',')).pop(-1)
        # X.append(map(float, vector.split(',')))
        # Y.append(X.pop(-1))
    print X[0], Y[0]
    fp.close()
