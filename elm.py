
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def input_to_hidden(x):
    a = np.dot(x, weight_in)
    return a
def predict(x):
    x = input_to_hidden(x)
    y = np.dot(x, weight_out)
    return y

if __name__=='__main__':


    dataset = pd.read_csv("../iris.csv")
    X = dataset.iloc[:, :4].values.astype('float')
    labels = dataset.iloc[:, 4].values
    for i in range(len(labels)):
        if labels[i] == 'Iris-setosa':
            labels[i] = 0
        elif labels[i] == 'Iris-versicolor':
            labels[i] = 1
        elif labels[i] == 'Iris-virginica':
            labels[i] = 2

    y_train = np.zeros([labels.shape[0], 3])

    for i in range(labels.shape[0]):
        y_train[i][labels[i]] = 1
    # print(len(y_train))
    # print(len(X))
    y_train.view(type=np.matrix)
    x_train, x_test, y_train, y_test = train_test_split(X, y_train, test_size=0.2)
    print(f'Train size: {x_train.shape[0]}, Test size: {x_test.shape[0]}')

    input_length = x_train.shape[1]
    hidden_units = 5
    weight_in = np.random.normal(size=[input_length, hidden_units])
    # print(weight_in.shape)
    # print(weight_in)
    # for i in weight_in:
    #   print(i)
    print(input_length)
    print('Input Weight shape: {shape}'.format(shape=weight_in.shape))
    X = input_to_hidden(x_train)
    print(X.shape)
    Xt = np.transpose(X)
    new = np.matrix(np.dot(Xt, X))
    # np.linalg.inv(new)
    weight_out = np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, y_train))
    print('Output weights shape: {shape}'.format(shape=weight_out.shape))
    y = predict(x_test)
    correct = 0
    total = y.shape[0]

    for i in range(total):
        predicted = np.argmax(y[i])
        test = np.argmax(y_test[i])
        correct = correct + (1 if predicted == test else 0)
    print('Accuracy: {:f}'.format(correct / total))


