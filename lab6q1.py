# Build a two-layer neural network for image recognition with the fashion MNIST data set. The
# network has one hidden layer, and an output layer. The hidden layer has 512 nodes, and adopts the ReLU
# activation function; the output layer has 10 nodes, and adopts the softmax activation function to output the class
# probabilities. Use the “sparse_categorical_crossentropy” loss function, use ‘adam’ as the optimizer, use a batch
# size of 32, and train your model for 5 epochs.

#The following code snippet is for you to start with:
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

import matplotlib.pyplot as plt
for i in range(10):
    plt.imshow(x_train[i,:,:], cmap='gray')
    plt.show()

x_train, x_test = x_train / 255.0, x_test / 255.0 # normalize the pixel values to be in [0, 1]
# note: you need to flatten the image to a vector, to serve as the input layer of the network.
# code to be implemented …
#28: image height, 28: image width
x_train = x_train.reshape(60000,784) #60000: number of training examples, 784: 28*28
x_test = x_test.reshape(10000,784) #10000: number of testing examples, 784: 28*28

model = keras.models.Sequential() #Sequential model is a linear stack of layers, creates a model to hold the layers
#Dense layer is a fully connected layer
model.add(keras.layers.Dense(512, activation='relu', input_shape=(784,))) #hidden layer, 512: number of neurons, activation function: relu, input_shape: 784
model.add(keras.layers.Dense(10, activation='softmax')) #output layer, 10: number of neurons, activation function: softmax

model.summary() #Prints a string summary of the network.

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])#Configures the model for training.

model.fit(x_train, y_train, epochs=5, batch_size=32)#Trains the model for a fixed number of epochs (iterations on a dataset).

test_loss, test_acc = model.evaluate(x_test, y_test)#Returns the loss value & metrics values for the model in test mode.

print("accuracy: ", test_acc)#Prints the accuracy of the model.

predicted_probability = model.predict(x_test)#Generates output predictions for the input samples.
y_test_hat = np.argmax(predicted_probability, axis=1)#Returns the indices of the maximum values along an axis.

#Compute confusion matrix to evaluate the accuracy of a classification.
#diagonal elements represent the number of elements that were predicted correctly
#vertical = ground truth labels
#horizontal = predicted categories
cm=confusion_matrix(y_test, y_test_hat, labels=range(10))
print(cm)

# Explain the source code: how to load dataset, build network layers, specify the loss function, batch size
# and epoch number, how to train the model, perform prediction on the test set, and generate the final class
# label for the test set?
# Answer:
#The dataset is loaded using the keras.datasets.fashion_mnist.load_data() function.
#The network layers are built using the keras.models.Sequential() function.
#The loss function is specified using the model.compile() function.
#The model is trained using the model.fit() function.
#The model is evaluated using the model.evaluate() function.
#The model is used to generate predictions using the model.predict() function.
#The final class label for the test set is generated using the np.argmax() function.


# Explain the confusion matrix. What do the diagonal elements of the matrix represent, and what do the offdiagonal elements of the matrix represent? From these results, do you think your neural network works
# well for this image recognition task?
# Answer:
#diagonal elements represent the number of elements that were predicted correctly
#vertical = ground truth labels
#horizontal = predicted categories
#The neural network works well for thixs image recognition task because the diagonal elements are much larger than the off-diagonal elements.
#The accuracy is 0.878, which is pretty good.
#predicted category: the category that the neural network predicts
#ground truth label: the actual category of the image
