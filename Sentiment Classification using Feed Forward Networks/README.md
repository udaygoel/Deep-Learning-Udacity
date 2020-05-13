# Sentiment Classification using Feed Forward Networks
This project follows the lessons taught by [Andrew Trask](http://iamtrask.github.io/) in [Udacity Deep Learning Nanodegree Program](https://www.udacity.com/course/deep-learning-nanodegree--nd101). It covers sentiment classification using feed forward neural networks using 25000 movie reviews. The network is trained on 24000 reviews and tested on 1000 reviews. The project is meant to illustrate the steps to improve classification and focusses on the techniques that can be applied. The network has not been optimized by fine tuning the hyperparameters.

### Project Files

The project uses these files:

- Sentiment_Classification.ipynb: This Jupyter Notebook covers the project. It starts with the analysis of the data and progressively builds the model for classification.
- reviews.txt: The 25000 movie reviews with reviews separated by a newline.
- labels.txt: The sentiment labels corresponding to the reviews.
- requirements.txt: python libraries to load in the python environment. 
- PNG Files: These are images used for illustration purposes in the Notebook

### Contents

There are 7 main sections of the Project.

1. Developing a Predictive Theory and Quick Theory Validation

   This section tests a prediction theory to split words into positive, negative and neutral sentiment. This is important to understand if there is any predictive power in our theory before we start building a neural network.

2. Creating the Input/Output Data

3. Building a Neural Network

   A feed forward neural network is built. Due to the simplicity of this project, the formulae have been written for forward and backward propagation, instead of using libraries such as Keras, Tensorflow or PyTorch. This is also run on the CPU. This model gives 69% accuracy on training data.

4. Reducing Noise in the Input Data

   The model is updated to reduce noise from common and frequently occurring words by removing the use of frequency of occurrence of each word and instead, only checking for the presence of each word. The accuracy on training data improves to 84.8% and the model achieves accuracy of 86% on test data.

5. Making Network more Efficient

   The input vector includes many zeros which leads to wasted compute time in multiplication and addition. Since the only values in the input vector are 0 and 1, the network is modified to simply sum up the weights of the first (and only) hidden layer where the input node has value of 1, to get the hidden layer output. Backpropagation is also optimised in similar manner. The training process speeds up by 10 times with no loss in accuracy.

6. Reduce Noise by Strategically Reducing the Vocabulary

   Two additional parameters, min_count and polarity_cutoff, are introduced to filter out rarely occurring words and neutral words, respectively. This results in much more faster model with similar accuracy. The improvement in speed depends on the number of words filtered out from the vocabulary. With 1 iteration, the improvement in training speed is another 10 times and the training accuracy is 83.3%. 

7. Finding Most Similar Words

   We can mathematically compute the similarity between two words using the cosine similarity. The project implements this by computing the dot product of the hidden layer input weights for those two words. We use this for illustration purposes, and so the implementation here doesn't divide by the product of the magnitude of these vectors. The vector T-SNE plot for most polarized words gives a visual representation of how the model is able to assign positive or negative sentiment to the words in the vocabulary.

   

   This projects explains the concepts that can be used for sentiment classification. The hyperparameters have not been fine tuned for maximum accuracy. 
