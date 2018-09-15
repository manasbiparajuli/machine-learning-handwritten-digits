import numpy as np 
import matplotlib.pyplot as plt 

# ## Test Accuracy vs Learning rate
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



## Test Accuracy vs Minibatch Size

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


####
# Test Accuracy vs Neuron size 
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



####
# Test Accuracy vs Learning Rate when using different activation functions

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