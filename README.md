# Deep-Learning-Udacity

Here you can find several projects covering various neural network architectures for deep learning applications. These projects were developed as part of the [Udacity Deep Learning Nanodegree Program](https://www.udacity.com/course/deep-learning-nanodegree--nd101). 



## Projects

You can view and launch the following projects:

- [Sentiment Classification using Feed Forward Networks](https://github.com/udaygoel/Deep-Learning-Udacity/tree/master/Sentiment%20Classification%20using%20Feed%20Forward%20Networks)

  This project develops a sentiment classification model using Feed Forward Networks. The Project mimics the lesson in the [Udacity Deep Learning Nanodegree Program](https://www.udacity.com/course/deep-learning-nanodegree--nd101) and runs through each step in detail. The model is trained on movie reviews and is able to achieve more than 80% accuracy.

- [Dog Breed Classifier - Convolutional Neural Networks](https://github.com/udaygoel/Deep-Learning-Udacity/tree/master/Dog%20Breed%20Classifier%20-%20Convolutional%20Neural%20Networks)

  This project develops a classifier to identify the dog breed using Convolutional Neural Networks. The network architecture applies transfer learning and uses the bottleneck features from Resnet 50 architecture in Keras to develop the classifier model. There are 133 dog breeds with 6680 training images, 835 validation images and 836 testing images. The project also uses the human face detection tool from OpenCV. Towards the end the classifier is run on a series of pictures collected from internet and identifies the breed of the dog.

- [TV Script Generation - LSTM Network](https://github.com/udaygoel/Deep-Learning-Udacity/tree/master/TV%20Script%20Generation%20-%20LSTM%20Network)

  This project uses a subset of [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts to train a LSTM neural network. This network is then used to generate a new TV Script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern). The input data is processed to create tokens, tokenize punctuations, vocabulary and create lookup tables to use for training the network. The network creates an embedding table using the vocabulary and the output of the embedding layer is fed as input to the LSTM network. The Project creates a new script that if fairly legible to read. This can be further improved by training on a bigger dataset. The hyperparameters can also be optimized depending on the output quality when trained on a different dataset.

