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
  
- [Face Generation - Generative Adversarial Networks](https://github.com/udaygoel/Deep-Learning-Udacity/tree/master/Face%20Generation%20-%20Generative%20Adversarial%20Networks)

  This project creates a Generative Adversarial Network (GAN) to generate new images of human faces. The GAN consists of a Generator and a Discriminator. The Generator creates new images using an input vector and the Discriminator compares the generated new image with the actual image. The network trains as the Generator learns to create new images (with no knowledge of actual image) that the Discriminator will accept as the true image, while the Discriminator learns to identify the new image as different from the actual image.  The GAN is first tested on [MNIST](http://yann.lecun.com/exdb/mnist/) data, which is a dataset containing images of handwritten digits, to see quickly how well the model trains. Once satisfied, the GAN is trained on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) data, which is a dataset of over 200,000 celebrity images with annotations. The trained GAN, in this project, is able to create images that all look like human faces.

- [Quadcopter - Reinforcement Learning](https://github.com/udaygoel/Deep-Learning-Udacity/tree/master/Quadcopter%20-%20Reinforcement%20Learning)

  This project applies Reinforcement Learning algorithm to design an agent to fly a quadcopter.  The implementation is inspired by the methodology in the original [Deep Deterministic Policy Gradient (DDPG) paper](https://arxiv.org/abs/1509.02971). This is based on the actor critic model where both are implemented using deep neural network. The task is to train the quadcopter to fly from an initial location to a target location. 

