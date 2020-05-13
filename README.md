# CS 1699
Course description: This course will cover the basics of modern deep neural networks. The first part of the course will introduce neural network architectures, activation functions, and operations. It will present different loss functions and describe how training is performed via backpropagation. In the second part, the course will describe specific types of neural networks, e.g. convolutional, recurrent, and graph networks, as well as their applications in computer vision and natural language processing. The course will also briefly discuss reinforcement learning and unsupervised learning, in the context of neural networks. In addition to attending lectures and completing bi-weekly homework assignments, students will also carry out and present a project.

Prerequisites: Math 220 (Calculus I), Math 280 or 1180 (Linear Algebra), CS 1501 (Algorithm Implementation)

Programming language/framework: We will use Python, NumPy/SciPy, and PyTorch.


## HW1
Various Numpy, Python, MatPlotLib exercises.

## HW2 
Built a feed-forward network by hand. The network has three layers and utilizes my own back-propagation algorithm. 
Network was trained on the [Red Wine Quality Dataset] (https://archive.ics.uci.edu/ml/datasets/Wine+Quality). 

## HW3
Used Pytorch to implement a 3-layer MLP on the CiFar-10 Dataset. Created a custom-fit dataset class for the Cifar data and used Pytorch's pretrained MobileNetV2 models for transfer learning.

## HW4
Implemented the original AlexNet CNN using PyTorch. Then created 4 varieties that used various pooling, filter, and kernel size strategies. Visualized each implentation's layers with the `visual_kernels` function.

## HW5
Implemented and compared four variants of a LSTM. Developed a character-based RNN model that predicts Shakespeare character and used the model to generate text. Visualized the LSTM's gates and internals using the `visualization.py` class.

## Project
Cancelled term project due to COVID-19. Our team aimed to generate music based on various moods.
