
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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
 
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

#######################################################################

# # Visualize the first digit of each class:

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train == i][0].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys')

# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# # plt.savefig('images/12_5.png', dpi=300)
# plt.show()


# # Visualize 25 different versions of "7":

# fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
# ax = ax.flatten()
# for i in range(25):
#     img = X_train[y_train == 7][i].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys')

# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# # plt.savefig('images/12_6.png', dpi=300)
# plt.show()


#######################################################################


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
from sklearn.model_selection import train_test_split



X_train, y_train, X_test, y_test = [mnist[f] for f in ['X_train', 'y_train', 
                                    'X_test', 'y_test']]

del mnist

X_train.shape
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# ## Implementing a multi-layer perceptron

# X_train=X_train[:55000]
# y_train=y_train[:55000]
# X_valid=X_train[55000:]
# y_valid=y_train[55000:]


# X_train=X_train[0:100]
# y_train=y_train[0:100]
# X_valid=X_train[100:250]
# y_valid=y_train[100:250]


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


# file = open("hidden_values.txt", "w")


# nn = NeuralNetMLP(n_hidden=100, 
#                 l2=0.01, 
#                 epochs=n_epochs, 
#                 eta=0.0005,
#                 minibatch_size=100, 
#                 shuffle=True,
#                 seed=1)

# nn.fit(X_train=X_train[:55000], 
#     y_train=y_train[:55000],
#     X_valid=X_train[55000:],
#     y_valid=y_train[55000:])

# nn.fit(X_train=X_train[0:100], 
#     y_train=y_train[0:100],
#     X_valid=X_train[100:250],
#     y_valid=y_train[100:250])


# y_test_pred = nn.predict(X_test)
# acc = (np.sum(y_test == y_test_pred)
#     .astype(np.float) / X_test.shape[0])

# print(' Test accuracy: %.2f%%' % (acc * 100))

# file.write (' Test accuracy: %.2f%%' % (acc * 100))
# file.write(str("\n"))

# file.close()

# ---
# **Note**
# 
# In the fit method of the MLP example above,
# 
# ```python
# 
# for idx in mini:
# ...
#     # compute gradient via backpropagation
#     grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
#                                       a3=a3, z2=z2,
#                                       y_enc=y_enc[:, idx],
#                                       w1=self.w1,
#                                       w2=self.w2)
# 
#     delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
#     self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
#     self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
#     delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
# ```
# 
# `delta_w1_prev` (same applies to `delta_w2_prev`) is a memory view on `delta_w1` via  
# 
# ```python
# delta_w1_prev = delta_w1
# ```
# on the last line. This could be problematic, since updating `delta_w1 = self.eta * grad1` would change `delta_w1_prev` as well when we iterate over the for loop. Note that this is not the case here, because we assign a new array to `delta_w1` in each iteration -- the gradient array times the learning rate:
# 
# ```python
# delta_w1 = self.eta * grad1
# ```
# 
# The assignment shown above leaves the `delta_w1_prev` pointing to the "old" `delta_w1` array. To illustrates this with a simple snippet, consider the following example:
# 
# 


# #####################################################

# import matplotlib.pyplot as plt

# plt.plot(range(nn.epochs), nn.eval_['cost'])
# plt.ylabel('Cost')
# plt.xlabel('Epochs')
# #plt.savefig('images/12_07.png', dpi=300)
# plt.show()


# # In[22]:


# plt.plot(range(nn.epochs), nn.eval_['train_acc'], 
#          label='training')
# plt.plot(range(nn.epochs), nn.eval_['valid_acc'], 
#          label='validation', linestyle='--')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.legend()
# #plt.savefig('images/12_08.png', dpi=300)
# plt.show()

# #######################################################################



# In[23]:


# y_test_pred = nn.predict(X_test)
# acc = (np.sum(y_test == y_test_pred)
#        .astype(np.float) / X_test.shape[0])

# print('Test accuracy: %.2f%%' % (acc * 100))


# In[24]:


# #######################################################################


# miscl_img = X_test[y_test != y_test_pred][:25]
# correct_lab = y_test[y_test != y_test_pred][:25]
# miscl_lab = y_test_pred[y_test != y_test_pred][:25]

# fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
# ax = ax.flatten()
# for i in range(25):
#     img = miscl_img[i].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#     ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# #plt.savefig('images/12_09.png', dpi=300)
# plt.show()

# #######################################################################