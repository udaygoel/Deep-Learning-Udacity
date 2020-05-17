# TV Script Generation - LSTM Network
This project uses a subset of [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts to train a LSTM neural network. This network is then used to generate a new TV Script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern). The input data is processed to create tokens, tokenize punctuations, vocabulary and create lookup tables to use for training the network. The network creates an embedding table using the vocabulary and the output of the embedding layer is fed as input to the LSTM network. The Project creates a new script that if fairly legible to read.

### Project Files

The project uses these files and folders:

- [dlnd_tv_script_generation.ipynb](https://github.com/udaygoel/Deep-Learning-Udacity/blob/master/TV%20Script%20Generation%20-%20LSTM%20Network/dlnd_tv_script_generation.ipynb): This Jupyter Notebook covers the project. It starts with the analysis and processing of the data and then progressively builds the LSTM based model. 
- helper.py: helper functions to process, load and save the data. This is provided by Udacity.
- problem_unittests.py: Unit tests to test the functions created in the notebook. This is provided by Udacity.
- preprocess.p: Saved file for preprocessed data. This data includes tokenized words, punctuation and the lookup table (together, the vocabulary).
- params.p: Saved file for the parameters of the input data. This is mainly the sequence length for this project.
- save.xxx files: There are 3 files with name as "save". These are the saved data of the trained model that includes the Embedding Layer and LSTM network.
- data/simpsons: This folder contains the data used to train the model.

### Contents

There are 8 main sections of the Project.

1. Importing Dataset and preprocessing to create the vocabulary items. These are saved in the preprocess.p file.

2. Building the Neural Network

   The Neural Network is created using TensorFlow. The input sentences are split into sequences with sequence length given by seq_length and batch size by batch_size. The neural network starts with an embedding layer that converts the input sequences into an embedding table. The output from embedding layer is fed to a LSTM layer, followed by a fully connected layer. The output size of the fully connected layer is same as the vocabulary size.

4. Training the Neural Network

   The neural network is trained on the GPU with 75 epochs. The loss function is the tensorflow.contrib.seq2seq.sequence_loss() function. The optimizer is AdamOptimizer. The gradients are clipped to prevent exploding gradients. The trained model is saved with the filenames starting with "save"

5. Generate TV Script

   The trained network is used to generate a TV script of 200 words. The starting word is "moe_szyslak". This script can be seen at the bottom of the notebook.


The TV Script generated is fairly legible to read. In order to get better results, the network can be trained on a bigger dataset. The hyperparameters can also be optimized depending on the output quality when trained on the bigger dataset.