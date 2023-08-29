# Problem 2: Implement an image compression system using a neural network. The image to be compressed is of
# size 𝑚 × 𝑛 (𝑚 rows and 𝑛 columns of pixels). Normalize the pixel values to be in the range of [0, 1].
# The structure of the neural network is the following:
# (1) The input layer is the flattened image, that is, a 1-dimensional vector with 𝑚 × 𝑛 elements
# (2) A compressed layer (hidden layer) with 𝑃 nodes, 𝑃 < 𝑚 × 𝑛, followed by ReLU activation
# (3) An expansion layer (hidden layer) with 𝑚 × 𝑛 × 𝑇 nodes, 𝑇 = 2 is the expansion factor, followed by
# ReLU activation
# (4) An output layer with 𝑚 × 𝑛 nodes, followed by Sigmoid activation
# (5) A reshape layer that convert the 1-dimensional vector output to the 𝑚 × 𝑛 2-dimensional image
# Use the same fashion MNIST data set from Problem 1 for this problem. Batch size=64. Epochs=10. The loss
# function (error function) is the mean-squared-error (mse) loss function. The mse is defined as
# mse =
# 1
# 𝑁
# ∑ (𝑥𝑖 − 𝑥̂𝑖)
# 𝑁 2
# 𝑖=1
# , where 𝑥𝑖
# is the 𝑖-th pixel value in the original (normalized) image, 𝑥̂𝑖
# is the 𝑖-th pixel
# value in the decoded (reconstructed) image, and 𝑁 is the number of pixels in one image.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import math

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

import matplotlib.pyplot as plt

for i in range(10):
    plt.imshow(x_test[i, :, :], cmap='gray')
    plt.show()

x_train, x_test = x_train / 255.0, x_test / 255.0 # normalize the pixel values to be in [0, 1]
#flatten the image to a vector, to serve as the input layer of the network using the reshape function.
#       784: 28*28
#       28: image height
#       28: image width

def modeler(p):
    model = keras.models.Sequential() #Sequential model is a linear stack of layers

    model.add(keras.layers.Flatten(input_shape = (28, 28))) #Flatten layer flattens the input.
    model.add(keras.layers.Dense(p, activation='relu')) #compressing the image
    model.add(keras.layers.Dense(28*28*2, activation='relu')) #encoding the image
    model.add(keras.layers.Dense(28*28, activation='sigmoid'))#decoding the image
    model.add(keras.layers.Reshape((28, 28))) #Reshape layer reshapes the input to the given shape.

    model.summary() #print the summary of the model

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])#Configures the model for training.

    model.fit(x_train, x_train, epochs=10, batch_size=64)#Trains the model for a fixed number of epochs (iterations on a dataset).

    output_images = model.predict(x_test)

    for i in range(10):
        plt.imshow(output_images[i, :, :], cmap='gray')
        plt.show()

    return output_images


# 2.a For three different 𝑃 values: 𝑃 = 10, 50, 200, train the network, and perform compression and decompression
# using the trained model on the test images. Calculate the average peak signal-to-noise ratio PSNR (in dB) value
# of the decompressed test frames versus the 𝑃 values. The PSNR is defined as: PSNR(dB) = 10log10 (
# 𝑀𝐴𝑋𝐼
# 2
# 𝑚𝑠𝑒 ),
# where 𝑀𝐴𝑋𝐼
# is the maximum pixel intensity value of the original image. In this experiment, for a normalized
# gray-scale image, 𝑀𝐴𝑋𝐼 = 1. Average PSNR: averaged over all test images.
# What’s the difference among the average PSNR of different 𝑃 values? What do you think is the reason of such a
# result?

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0): #in case of divided by 0
        return 100
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr
model1 = modeler(10)
model2 = modeler(50)
model3 = modeler(200)
print("psnr, p=10", psnr(x_test, model1))
print("psnr, p=50",psnr(x_test, model2))
print("psnr, p=200",psnr(x_test, model3))

def average_psnr(original, compressed):
    sum = 0
    for i in range(10000):
        sum += psnr(original[i], compressed[i])
    return sum/10000

print("average psnr, p=10", average_psnr(x_test, model1))
print("average psnr, p=50", average_psnr(x_test, model2))
print("average psnr, p=200", average_psnr(x_test, model3))

# What’s the difference among the average PSNR of different 𝑃 values? What do you think is the reason of such a
# result?
# The average PSNR of different 𝑃 values are 19.7, 22.7, 22.4 respectively. The reason is that the more nodes in the
# compressed layer, the more information can be stored, and the more accurate the reconstruction is.


# 2.b In one figure, display the first 10 test images and their decompressed images with 𝑃 = 10, 50, and 200 in four
# rows: (a) the original 10 images, (b) the corresponding decompressed images with 𝑃 = 10, (c) the decompressed
# images with 𝑃 = 50, and (d) the decompressed images with 𝑃 = 200.
def plotter(model):
    output_images = model
    for i in range(10):
        plt.subplot(4, 10, i + 1)
        plt.imshow(x_test[i, :, :], cmap='gray')
        plt.subplot(4, 10, i + 11)
        plt.imshow(output_images[i, :, :], cmap='gray')
        plt.subplot(4, 10, i + 21)
        plt.imshow(output_images[i + 10, :, :], cmap='gray')
        plt.subplot(4, 10, i + 31)
        plt.imshow(output_images[i + 20, :, :], cmap='gray')
    plt.show()

plotter(model1)
plotter(model2)
plotter(model3)

#What do you observe from the decompressed images (the visual quality of the decompressed images of different
#𝑃 values)?
# The decompressed images with 𝑃 = 200 are the most similar to the original images, and the decompressed images
# with 𝑃 = 10 are the most different from the original images.


#for demo:
# Explain the source code: how to build network layers, specify the loss function, batch size and epoch
# number, how to calculate the average PSNR for the test set?
#answer:
# • How to build network layers: Sequential model is a linear stack of layers. The first layer is Flatten layer, which
# is used to flatten the input. The second layer is Dense layer, which is used to connect all the input nodes to all
# the output nodes. The third layer is Dense layer, which is used to connect all the input nodes to all the output
# nodes. The fourth layer is Dense layer, which is used to connect all the input nodes to all the output nodes. The
# fifth layer is Reshape layer, which is used to reshape the input to the specified shape.
# The average PSNR is calculated by the function average_psnr(original, compressed), which is defined in the
# previous question.

# • Which layers of the network belong to the encoder (compression), which layers of the network belong to
# the decoder (decompression)?
# Compress layer: Flatten layer, Compress ,Encode layer, Decode layer, Shaping layer
# • For different 𝑃 values, what are the quality of the decoded images, and why?
# The quality of the decoded images is different. The reason is that the more nodes in the compressed layer, the
# more information can be stored, and the more accurate the reconstruction is.





