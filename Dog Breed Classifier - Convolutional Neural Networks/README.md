# Dog Breed Classifier - Convolutional Neural Networks
This project develops a classifier to identify the dog breed using Convolutional Neural Networks. The network architecture applies transfer learning and uses the bottleneck features from Resnet 50 architecture in Keras to develop the classifier model. There are 133 dog breeds with 6680 training images, 835 validation images and 836 testing images. The project also uses the human face detection tool from OpenCV. Towards the end, the classifier is run on a series of pictures collected from internet to identify whether it is a human or a dog. In case of dog, it predicts the breed of the dog. For human, it predicts the dog breed the face resembles. For others, it simply says the face is neither human nor dog. The classifier achieves a high accuracy score in this prediction task.

### Project Files

The project uses these files and folders:

- dog_app.ipynb: This Jupyter Notebook covers the project. It starts with the analysis of the data and progressively builds the model for classification. It tests different models before building the final chosen model.
- extract_bottleneck_features.py: This script has the helper functions to retrieve the bottleneck features for some well known pre-trained models such as Resnet 50, VGG-16
- saved_models: This folders stores the classifiers created in the notebook.
- Step7 images: This folder stores the images used in the prediction task at the end of the book.
- haarcascades: We use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images. OpenCV provides many pre-trained detectors and one of these detectors has been stored in this folder
- images: This folder stores the images generally used in the notebook. These are not used for any analysis or by any model.

The repository does not include these files due to space constraints:

- app/dog_images: These are the images used for training, validation and testing of the classifier.
- app/lfw: These are the images of the human faces, used randomly in the notebook to illustrate the use of the face detector.
- app/bottleneck_features: These are bottleneck features created for the dog images in app/dog_images. Although, these have been mentioned in the notebook, they have not been used. The features were created for Resnet 50 using a previous implementation in Keras and this had the bottleneck size of 1 x 1. Since Keras v 2.2.0, the implementation produces bottleneck size of 7 x 7. Details can be seen [here](https://github.com/keras-team/keras-applications/issues/50). As a result, these features are re-created in the notebook.

### Contents

There are 8 main sections of the Project.

1. Importing Datasets and creating training, validation and testing sets.

2. Detect Humans

   In this section, we load the detector stored in the haarcascades directory and demonstrate its uses on a sample image

3. Detect dogs

   We use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images. This model is trained on ImageNet that contains over 10 million images, each containing an object from one of 1000 categories. Of these 1000 categories, there are 133 categories of dog breeds. The model is downloaded using Keras, images are pre-processed and a dog detector function is created.

4. Create a CNN to classify Dog Breeds

   A CNN classifier is created from scratch using combinations of Convolution layers, Max Pooling layers, Global Average Pooling layers, Dropout layers and finally, a Dense layer. This is implemented using Keras. Activation functions ReLu and Softmax are used. The output size is 133 to match the number of dog breeds in our data set. The data set is the processed images from app/dog_images.  The model is trained on 40 epochs with early stopping mechanism. The best model by validation loss is stored [here](https://github.com/udaygoel/Deep-Learning-Udacity/tree/master/Dog%20Breed%20Classifier%20-%20Convolutional%20Neural%20Networks/saved_models) as weights.best.from_scratch.hdf5. This model gets a testing accuracy of only 8.7%.

5. Using a CNN to classify Dog Breeds (using Transfer Learning)

   In this section, we create a model using the pre-trained VGG-16 model as a fixed feature extractor. The last convolutional output of VGG-16 is fed as input to a global average pooling layer and a fully connected layer. The output size is 133. Running the same training as earlier section, the model gets a testing accuracy of 74%. The best model by validation loss is stored [here](https://github.com/udaygoel/Deep-Learning-Udacity/tree/master/Dog%20Breed%20Classifier%20-%20Convolutional%20Neural%20Networks/saved_models) as weights.best.VGG16.hdf5.

6. Creating a CNN to classify Dog Breeds (using Transfer Learning)

   In this section, we create a model using the pre-trained Resnet-50 model as a fixed feature extractor. The last convolutional output of Resnet-50 model is fed as input to a global average pooling layer and a fully connected layer. The output size is 133. Running the same training as earlier section, the model gets a testing accuracy of 82%. The best model by validation loss is stored [here](https://github.com/udaygoel/Deep-Learning-Udacity/tree/master/Dog%20Breed%20Classifier%20-%20Convolutional%20Neural%20Networks/saved_models) as weights.best.Resnet50.hdf5.

7. Algorithm Design

   The algorithm accepts a file path to an image and first determines whether the image contains a human, dog or neither. Then

   - If a **dog** is detected in the image, return the predicted breed
   - If a **human** is detected in the image, return the resembling dog breed
   - If **neither** is detected in the image, provide output that indicates an error

   The algorithm uses the classifier designed in Step 2 to detect a human face, Step 3 to detect a dog and Step 6 to return the predicted breed.

8. Testing the Algorithm

   This is the most interesting part of the project. Here, we get to test the algorithm against a range of images. These images are not part of the dataset used to train, validate or test the classifiers. The images were taken from internet and are stored in [Step7 Images](https://github.com/udaygoel/Deep-Learning-Udacity/tree/master/Dog%20Breed%20Classifier%20-%20Convolutional%20Neural%20Networks/Step7%20Images) folder. In the notebook, we use 36 images. 

   

   The results from testing the algorithm are as follows:

   - **Dog** images: 25 images. 
     - 24 are correctly identified as a dog. The breed for 19 dogs is correctly identified.
   - **Human** images: 3 images
     - 2 are correctly identified. The one not identified has the person (Thor) wearing a helmet that hides some facial features.
   - **Neither** human nor dog: 8 images
     - 6 are correctly identified as neither human nor dog. The 2 that are incorrectly identified are both fox. These are the only images for fox. It appears that the human face detector is identifying fox as a human.
