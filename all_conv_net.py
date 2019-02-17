
import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.autograd import Variable
import numpy as np
#import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.nn import Sequential
import torch.nn as nn
import random

print("7:22 pm")

data = np.load("train_feats.npy")
test_data = np.load("test_feats.npy")
train_labels = np.load("train_labels.npy")

# Considering small amount of data for coputational efficiency
data = data[:50000]
test_data = test_data[:len(test_data)]
train_labels = train_labels[:50000]

## Shuffling the data by index
#index = np.arange(len(data))
#np.random.shuffle(index)
#data = data[index]
##test_data = test_data[index]
#train_labels = train_labels[index]

def sample_zero_mean(x):

    for i in range(0,len(x)):
        mu = np.sum(x[i]) / (np.size(x[i]))
        x[i] = x[i] - mu

    return x

def gcn(x, scale=55., bias=0.01):

    for i in range(0,len(x)):
        mu = np.sum(x[i]) / (np.size(x[i]))
        sigma = np.sum((x[i] - mu)**2) / np.size(x[i])
        x[i] = scale * x[i] / np.sqrt(bias + sigma)

    return x

def feature_zero_mean(x, xtest):

    means = np.mean(x, axis = 0)
    x_r = []
    xtest_r = []
    for i in range(0,len(x.T)):
        temp = x[:, i] - means[i]
        x_r.append(temp)
        temp2 = xtest[:, i] - means[i]
        xtest_r.append(temp2)
    x_r = np.array(x_r)
    xtest_r = np.array(xtest_r)
    return (x_r.T, xtest_r.T)

def zca(x, xtest, bias=0.1):

    U,S,V = np.linalg.svd(np.dot((x.T), x) / len(x) + np.eye(len(x.T)) * bias)
    pca1 = np.dot(U, np.diag(1/np.sqrt(S)))
    pca = np.dot(pca1, U.T)
    #print(np.shape(pca))
    zca_x = np.dot(x, pca)
    zca_x_test = np.dot(xtest, pca)
    return (zca_x, zca_x_test)


def cifar_10_preprocess(x, xtest, image_size=32):

    x_r = sample_zero_mean(x)
    xtest_r = sample_zero_mean(xtest)

    x_r = gcn(x_r)
    xtest_r = gcn(xtest_r)

    x_r1, xtest_r1 = feature_zero_mean(x_r, xtest_r)

    x_r2, xtest_r2 = zca(x_r1, xtest_r1)


    x_r3  = np.reshape(x_r2, (len(x_r2), 3, image_size, image_size))
    xtest_r3  = np.reshape(xtest_r2, (len(xtest_r2), 3, image_size, image_size))

    return(x_r3, xtest_r3)

x_train, x_test = cifar_10_preprocess(data, test_data, 32)

x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()
y_train = torch.from_numpy(train_labels).long()

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

def all_cnn_module():

    model = Sequential(*[
            # Input Dropout
            nn.Dropout(0.2),
            # First Stage
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1,stride = 1),
            nn.ReLU(),
            # Second Stage
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, stride = 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Third Stage
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, stride = 1),
            nn.ReLU(),
            # Fourth Stage
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, stride = 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Fifth Stage
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=0, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, padding=0, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=10, kernel_size=1, padding=0, stride = 1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 6),
            Flatten()]
            )
    return model

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()

net = all_cnn_module()
net.apply(init_weights)

x_train = x_train.numpy()
x_test = x_test.numpy()
y_train = y_train.numpy()

n_splits = 100
x_train = np.split(x_train, n_splits)
y_train = np.split(y_train, n_splits)
x_test = np.split(x_test, n_splits)


index = np.arange(n_splits)
def training_routine(x_train, x_test, y_train, idn_, net):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)

    for i in range(40):
        j = 0

        random.shuffle(idn_)

        idn = idn_

        while(j<len(x_train)):

            train_batch = x_train[idn[j]]
            pred_batch = y_train[idn[j]]
            train_batch = torch.from_numpy(train_batch).float()
            pred_batch = torch.from_numpy(pred_batch).long()

            gpu = True
            if gpu:
                    train_batch = train_batch.cuda()
                    pred_batch = pred_batch.cuda()
                    net = net.cuda()

            y_output = net(train_batch)

            if (idn[j] == 27):
                print(idn[j])
                y_output_ = net(train_batch)
                pred_batch_ = pred_batch
                train_loss_ = criterion(y_output_, pred_batch_)
            train_loss = criterion(y_output, pred_batch)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(idn[j])
            j = j+1

        if i%1 == 0:
#            y_output = net(train_batch)
            train_prediction = y_output_.cpu().detach().argmax(dim=1)
            train_accuracy = (train_prediction.cpu().numpy()==pred_batch_.cpu().numpy()).mean()
            print('Training accuracy is {} after {} iterations'.format(train_accuracy, i+1))
            print('Loss is {} after {} iterations'.format(train_loss_, i+1))

    print("Testing")
    k = 0
    f1 = open('op.txt', 'w')
    file_predict = []
    while(k<len(x_test)):

        test_batch = x_test[k]
        test_batch = torch.from_numpy(test_batch).float()

        gpu = True
        if gpu:
                test_batch = test_batch.cuda()

        y_output_test = net(test_batch)
        test_pred = y_output_test.cpu().detach().argmax(dim=1)
        file_predict.append((test_pred.numpy()))
        for count in range(len(test_pred.numpy())):
            f1.write('{} \n'.format(test_pred.numpy()[count]))

        #print('\n')
        k = k + 1

    return file_predict

some = training_routine(x_train, x_test, y_train, index, net)

thing = np.reshape(some, [len(test_data), 1])



def write_results(predictions, output_file='predictions.txt'):

    with open(output_file, 'w') as f:
        for y in predictions:
            f.write("{}\n".format(y[0]))

write_results(thing, output_file='predictions.txt')
