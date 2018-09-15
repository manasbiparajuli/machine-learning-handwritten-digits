
# coding: utf-8

# *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Python Machine Learning - Code Examples

# # Chapter 12 - Implementing a Multi-layer Artificial Neural Network from Scratch
# 


# ## Obtaining the MNIST dataset

# The MNIST dataset is publicly available at http://yann.lecun.com/exdb/mnist/ and consists of the following four parts:
# 
# - Training set images: train-images-idx3-ubyte.gz (9.9 MB, 47 MB unzipped, 60,000 samples)
# - Training set labels: train-labels-idx1-ubyte.gz (29 KB, 60 KB unzipped, 60,000 labels)
# - Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 7.8 MB, 10,000 samples)
# - Test set labels: t10k-labels-idx1-ubyte.gz (5 KB, 10 KB unzipped, 10,000 labels)
# 
# In this section, we will only be working with a subset of MNIST, thus, we only need to download the training set images and training set labels. After downloading the files, I recommend unzipping the files using the Unix/Linux gzip tool from the terminal for efficiency, e.g., using the command 
# 
#     gzip *ubyte.gz -d
#  
# in your local MNIST download directory, or, using your favorite unzipping tool if you are working with a machine running on Microsoft Windows. The images are stored in byte form, and using the following function, we will read them into NumPy arrays that we will use to train our MLP.
# 


import os
import struct
import numpy as np
 
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, 
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, 
                               '%s-images-idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
 
    return images, labels



# unzips mnist

import sys
import gzip
import shutil

if (sys.version_info > (3, 0)):
    writemode = 'wb'
else:
    writemode = 'w'

zipped_mnist = [f for f in os.listdir('./') if f.endswith('ubyte.gz')]
for z in zipped_mnist:
    with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
        outfile.write(decompressed.read()) 


# Load training records
X_train, y_train = load_mnist('', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))


# Load testing records
X_test, y_test = load_mnist('', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# To save and restore - optional
np.savez_compressed('mnist_scaled.npz', 
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test)


mnist = np.load('mnist_scaled.npz')
mnist.files


# In[16]:


X_train, y_train, X_test, y_test = [mnist[f] for f in ['X_train', 'y_train', 
                                    'X_test', 'y_test']]

del mnist

X_train.shape
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# ## Implementing a multi-layer perceptron


class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ------------
    n_hidden : int (default: 30)
        Number of hidden units.
    l2 : float (default: 0.)
        Lambda value for L2-regularization.
        No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatche_size : int (default: 1)
        Number of training samples per minibatch.
    seed : int (default: None)
        Random seed for initalizing weights and shuffling.

    Attributes
    -----------
    eval_ : dict
      Dictionary collecting the cost, training accuracy,
      and validation accuracy for each epoch during training.

    """
    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """Encode labels into one-hot representation

        Parameters
        ------------
        y : array, shape = [n_samples]
            Target values.

        Returns
        -----------
        onehot : array, shape = (n_samples, n_labels)

        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))


    def _tanh(self, z):
        e_p = np.exp(z)
        e_m = np.exp(-z)
        return (e_p - e_m) / (e_p + e_m)

    def _forward(self, X):
        """Compute forward propagation step"""

        # step 1: net input of hidden layer
        # [n_samples, n_features] dot [n_features, n_hidden]
        # -> [n_samples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h

        # step 2: activation of hidden layer
        a_h = self._sigmoid(z_h)

        # step 3: net input of output layer
        # [n_samples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_samples, n_classlabels]

        z_out = np.dot(a_h, self.w_out) + self.b_out

        # step 4: activation output layer
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """Compute cost function.

        Parameters
        ----------
        y_enc : array, shape = (n_samples, n_labels)
            one-hot encoded class labels.
        output : array, shape = [n_samples, n_output_units]
            Activation of the output layer (forward propagation)

        Returns
        ---------
        cost : float
            Regularized cost

        """
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_samples]
            Predicted class labels.

        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """ Learn weights from training data.

        Parameters
        -----------
        X_train : array, shape = [n_samples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_samples]
            Target class labels.
        X_valid : array, shape = [n_samples, n_features]
            Sample features for validation during training
        y_valid : array, shape = [n_samples]
            Sample labels for validation during training

        Returns:
        ----------
        self

        """
        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]

        ########################
        # Weight initialization
        ########################

        # weights for input -> hidden
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        # weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs))  # for progress formatting
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):

            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                ##################
                # Backpropagation
                ##################

                # [n_samples, n_classlabels]
                sigma_out = a_out - y_train_enc[batch_idx]

                # [n_samples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_samples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_samples, n_hidden]
                sigma_h = (np.dot(sigma_out, self.w_out.T) *
                           sigmoid_derivative_h)

                # [n_features, n_samples] dot [n_samples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)

                # [n_hidden, n_samples] dot [n_samples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)

                # Regularization and weight updates
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            #############
            # Evaluation
            #############

            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self._forward(X_train)
            
            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100, valid_acc*100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self



n_epochs = 200


###########################################
#### Learning Rate
###########################################
eta_values = [ .001, 0.002, 0.003, 0.0045, 0.006, 0.009, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.00075, 0.00015, 0.00025, 0.00035, 0.00045, 0.00055, 0.00065, 0.0007, 0.00085, 0.0009, 0.00001, 0.000015, 0.00002, 0.000025]
file = open("eta_values.txt", "a")

for val in eta_values:

    nn = NeuralNetMLP(n_hidden=100, 
                    l2=0.01, 
                    epochs=n_epochs, 
                    eta=val,
                    minibatch_size=100, 
                    shuffle=True,
                    seed=1)

    print ("eta: ", val)
    file.write("eta: " + str(val))

    nn.fit(X_train=X_train[:55000], 
        y_train=y_train[:55000],
        X_valid=X_train[55000:],
        y_valid=y_train[55000:])

    y_test_pred = nn.predict(X_test)
    acc = (np.sum(y_test == y_test_pred)
       .astype(np.float) / X_test.shape[0])

    print(' Test accuracy: %.2f%%' % (acc * 100))
    file.write (' Test accuracy: %.2f%%' % (acc * 100))
    file.write(str("\n"))

file.close()


###########################################
###########################################

##########################################
## Minibatch Size
###########################################

n_epochs = 200

minibatch_values = [ 20, 35, 50, 75, 90, 100, 150, 200, 260, 310, 350, 420, 500, 550, 700, 30, 40, 60, 70, 80, 125, 175, 190, 220, 240, 270, 300, 380, 400, 450, 475, 525, 540, 575, 600, 625, 650, 750, 800, 850, 900, 950, 1000]
file = open("minibatch_values.txt", "a")

for val in minibatch_values:

    nn = NeuralNetMLP(n_hidden=100, 
                    l2=0.01, 
                    epochs=n_epochs, 
                    eta=0.0005,
                    minibatch_size=val, 
                    shuffle=True,
                    seed=1)

    print ("minibatch_size: ", val)
    file.write("minibatch_size: " + str(val))

    nn.fit(X_train=X_train[:55000], 
        y_train=y_train[:55000],
        X_valid=X_train[55000:],
        y_valid=y_train[55000:])

    y_test_pred = nn.predict(X_test)
    acc = (np.sum(y_test == y_test_pred)
       .astype(np.float) / X_test.shape[0])

    print(' Test accuracy: %.2f%%' % (acc * 100))
    file.write (' Test accuracy: %.2f%%' % (acc * 100))
    file.write(str("\n"))

file.close()

###########################################
###########################################


###########################################
## Number of neurons in first layer
###########################################
n_epochs = 200

neurons_values = [ 20, 40, 50, 75, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 120, 140, 170, 190, 225, 280, 320, 340, 375, 420, 440, 470, 490, 520, 540, 570]
file = open("neurons_values.txt", "a")

for val in neurons_values:

    nn = NeuralNetMLP(n_hidden=val, 
                    l2=0.01, 
                    epochs=n_epochs, 
                    eta=0.0005,
                    minibatch_size=100, 
                    shuffle=True,
                    seed=1)

    print ("neurons_size: ", val)
    file.write("neurons_size: " + str(val))

    nn.fit(X_train=X_train[:55000], 
        y_train=y_train[:55000],
        X_valid=X_train[55000:],
        y_valid=y_train[55000:])

    y_test_pred = nn.predict(X_test)
    acc = (np.sum(y_test == y_test_pred)
       .astype(np.float) / X_test.shape[0])

    print(' Test accuracy: %.2f%%' % (acc * 100))
    file.write (' Test accuracy: %.2f%%' % (acc * 100))
    file.write(str("\n"))

file.close()



##############################################
## Tanh Function and learning rate using scikit learn's MLPClassifier
##############################################
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


eta_values = [ .001, 0.002, 0.003, 0.0045, 0.006, 0.009, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.00075, 0.00015, 0.00025, 0.00035, 0.00045, 0.00055, 0.00065, 0.0007, 0.00085, 0.0009, 0.00001, 0.000015, 0.00002, 0.000025]

file = open("tanh.txt","a")

for val in eta_values:

    mlp = MLPClassifier (hidden_layer_sizes=100,
                        activation='tanh',
                        max_iter=200,
                        alpha=0.01,
                        solver='adam',
                        batch_size=100,
                        verbose=10,
                        shuffle=True,
                        random_state=None,
                        learning_rate_init=val,
                        early_stopping=False)

    mlp.fit(X_train, y_train)
    y_test_pred = mlp.predict(X_test)

    file.write("eta: " + str(val))

    print ("Training set score: ", accuracy_score(y_train, mlp.predict(X_train)))
    print ("Test set score: ", accuracy_score(y_test, y_test_pred))

    file.write("Test score: " + str(accuracy_score(y_test, y_test_pred)))
    file.write(str("\n"))

file.close()

##############################################
##############################################



##############################################
## Adding new hidden layer and modeling using scikit's MLPClassifier
##############################################

# Array of number of neurons in first and second hidden layer
array = [ [100, 50], [150, 100], [200, 150], [250, 200], [300, 250], [350, 300], [400, 350], [450, 400], [500, 450], [550, 500]]

file = open("hidden.txt", "a")

for val in array:

    mlp = MLPClassifier (hidden_layer_sizes=val,
                        activation='logistic',
                        max_iter=200,
                        alpha=0.01,
                        solver='adam',
                        batch_size=100,
                        verbose=10,
                        shuffle=True,
                        random_state=None,
                        learning_rate_init=0.0005,
                        early_stopping=True)

    mlp.fit(X_train, y_train)
    y_test_pred = mlp.predict(X_test)

    # print ("Training set score: ", accuracy_score(y_train, mlp.predict(X_train)))
    # print ("Test set score: ", accuracy_score(y_test, y_test_pred))

    file.write("hidden: " + str(val))
    file.write("Test accuracy: " + str(accuracy_score(y_test, y_test_pred)))
    file.write(str("\n"))

file.close()

##############################################
##############################################



##############################################
## Plotting our observations using matplotlib
## The pair of values are listed from the text files where we wrote our results into
##############################################


import matplotlib.pyplot as plt 

##############################################
# ## Test Accuracy vs Learning rate
##############################################
eta_values = [ [.001, 97.86], [0.002, 97.89],  [0.003,97.92], [0.0045,97.8], [0.006, 97.69], [0.009, 95.53], [0.0001, 96.10], [0.0002, 96.92], [0.0003, 97.31], [0.0004, 97.42], [0.0005, 97.54], [0.0006, 97.72], [0.00075, 97.78], [0.00015, 96.61], [0.00025, 97.17], [0.00035, 97.37], [0.00045, 97.43], [0.00055, 97.63], [0.00065, 97.76], [0.0007, 97.77], [0.00085, 97.86], [0.0009, 97.87], [0.00001, 91.28], [0.000015, 91.93], [0.00002, 92.57], [0.000025, 93.18]]

eta_values = sorted(eta_values, key=lambda x: x[0])

x = [i[0] for i in eta_values]
y = [i[1] for i in eta_values]

plt.subplots(1, 1, figsize=(10,10))
plt.plot(x, y, 'o--', markerfacecolor='red', markersize=5)

max_y_indice = y.index(max(y))
max_y = y[max_y_indice]
max_x = x[max_y_indice]

min_y_indice = y.index(min(y))
min_y = y[min_y_indice]
min_x = x[min_y_indice]

plt.annotate(s="Minimum Accuracy (" + str(min_x)+ "," + str(min_y)+ ")", xy= [min_x, min_y], fontsize=10)
plt.annotate(s="Maximum Accuracy (" + str(max_x)+ ',' + str(max_y)+')', xy= [max_x, max_y], fontsize=10)

plt.xlabel ("Learning Rate", fontsize=30)
plt.ylabel ("Test Accuracy", fontsize= 30)
plt.title ("Test Accuracy vs Learning Rate", fontsize=30)
# plt.show()
plt.savefig("learning_rate_sigmoid.png", dpi=300)


##############################################
## Test Accuracy vs Minibatch Size
##############################################

minibatch_values = [ [20, 97.24],  [35, 97.56], [50, 97.52], [75, 97.56], [90, 97.52], [100, 97.54], [150, 97.51], [200, 97.54], [260, 97.63], [310, 97.53], [350, 97.55], [420, 97.53], [500, 97.63], [550, 97.66], [700, 97.46], [30, 97.51], [40, 97.54], [60, 97.51], [70, 97.5], [80, 97.49], [125, 97.51], [175, 97.55], [190, 97.61], [220, 97.62], [240, 97.6], [270, 97.61], [300, 97.51], [380, 97.52], [400, 97.53], [450, 97.5], [475, 97.68], [525, 97.67], [540, 97.62], [575, 97.62], [600, 97.61], [625, 97.69], [650, 97.66], [750, 97.44], [800, 97.54], [850, 97.48], [900, 97.61], [950, 97.52], [1000, 97.45]]

minibatch_values = sorted(minibatch_values, key=lambda x: x[0])

x = [i[0] for i in minibatch_values]
y = [i[1] for i in minibatch_values]

plt.subplots(1, 1, figsize=(10,10))
plt.plot(x, y, 'o--', markerfacecolor='red', markersize=5)

max_y_indice = y.index(max(y))
max_y = y[max_y_indice]
max_x = x[max_y_indice]

min_y_indice = y.index(min(y))
min_y = y[min_y_indice]
min_x = x[min_y_indice]

plt.annotate(s="Minimum Accuracy (" + str(min_x)+ "," + str(min_y)+ ")", xy= [min_x, min_y], fontsize=10)
plt.annotate(s="Maximum Accuracy (" + str(max_x)+ ',' + str(max_y)+')', xy= [max_x, max_y], fontsize=10)

plt.xlabel ("Minibatch Size", fontsize=30)
plt.ylabel ("Test Accuracy", fontsize= 30)
plt.title ("Test Accuracy vs Minibatch Size", fontsize=30)
# plt.show()
plt.savefig("minibatch_size.png", dpi=300)


##############################################
# Test Accuracy vs Neuron size 
##############################################

neurons_values = [ [20, 95.15], [40, 96.74],  [50, 96.81], [75, 97.37], [90, 97.54], [100, 97.54], [150, 97.93], [200, 98.02], [250, 97.90], [300, 98], [350, 98.03], [400, 98.18], [450, 98.1], [500, 98.2], [550, 98.33], [600, 98.25], [120, 97.62], [140, 97.95], [170, 97.83], [190, 97.97], [225, 98.09], [280, 98.02], [320, 98.21], [340, 98.13], [375, 98.08], [420, 98.21], [440, 98.13], [470, 98.12], [490, 98.2], [520, 98.13], [540, 98.24], [570, 98.22]]

neurons_values = sorted(neurons_values, key=lambda x: x[0])

x = [i[0] for i in neurons_values]
y = [i[1] for i in neurons_values]

plt.subplots(1, 1, figsize=(15,15))
plt.plot(x, y, 'o--', markerfacecolor='red', markersize=5)

max_y_indice = y.index(max(y))
max_y = y[max_y_indice]
max_x = x[max_y_indice]

min_y_indice = y.index(min(y))
min_y = y[min_y_indice]
min_x = x[min_y_indice]

plt.annotate(s="Minimum Accuracy (" + str(min_x)+ "," + str(min_y)+ ")", xy= [min_x, min_y], fontsize=10)
plt.annotate(s="Maximum Accuracy (" + str(max_x)+ ',' + str(max_y)+')', xy= [max_x, max_y], fontsize=10)

plt.xlabel ("Neurons Size", fontsize=30)
plt.ylabel ("Test Accuracy", fontsize= 30)
plt.title ("Test Accuracy vs Neurons Size", fontsize=30)
# plt.show()
plt.savefig("neurons_size.png", dpi=300)



##############################################
# Test Accuracy vs Learning Rate when using different activation functions
##############################################

# Eta values using tanh function
eta_values_tanh = [ [.001, 97.50], [0.002, 95.38],  [0.003,93.94], [0.0045,93.07], [0.006, 89.48], [0.009, 88.80], [0.0001, 97.67], [0.0002, 97.77], [0.0003, 97.74], [0.0004, 97.48], [0.0005, 97.38], [0.0006, 97.56], [0.00075, 97.14], [0.00015, 97.72], [0.00025, 97.67], [0.00035, 97.55], [0.00045, 97.51], [0.00055, 97.69], [0.00065, 97.25], [0.0007, 97.23], [0.00085, 97.30], [0.0009, 97.36], [0.00001, 97], [0.000015, 97.27], [0.00002, 97.47], [0.000025, 97.48]]

eta_values_tanh = sorted(eta_values_tanh, key=lambda x: x[0])

x = [i[0] for i in eta_values_tanh]
y = [i[1] for i in eta_values_tanh]

plt.subplots(1, 1, figsize=(30,30))
plt.plot(x, y, 'o--', markerfacecolor='green', markersize=5)

max_y_indice = y.index(max(y))
max_y = y[max_y_indice]
max_x = x[max_y_indice]

min_y_indice = y.index(min(y))
min_y = y[min_y_indice]
min_x = x[min_y_indice]

plt.annotate(s="Minimum Accuracy (" + str(min_x)+ "," + str(min_y)+ ")", xy= [min_x, min_y], fontsize=25)
plt.annotate(s="Maximum Accuracy (" + str(max_x)+ ',' + str(max_y)+')', xy= [max_x, max_y], fontsize=25)

plt.xlabel ("Learning Rate", fontsize=25)
plt.ylabel ("Test Accuracy", fontsize= 25)
plt.title ("Test Accuracy vs Learning Rate Using Tanh Activation Function", fontsize=25)
# plt.show()
plt.savefig("learning_rate_tanh.png", dpi=300)