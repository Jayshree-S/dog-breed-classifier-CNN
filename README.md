## Dog-breed-classifier using CNN and transfer learning in PyTorch
This is the repo of Dog breed classifier project in Udacity ML Nanodegree. 

### Project Overview
<p align="justify">
The goal of the project is to build a machine learning model that can be used within
web app to process real-world, user-supplied images. The algorithm has to perform
two tasks:</p>
<ul>
<li>Given an image of a dog, the algorithm will identify an estimate
of the canine’s breed.</li>
<li>If supplied an image of a human, the code will identify the
resembling dog breed.</li>
</ul>
<p align="justify">For performing this multiclass classification, we can use <b>Convolutional Neural Network</b> to solve the problem.The solution involves three steps. First, to detect
human images, we can use existing algorithm like OpenCV’s implementation of
Haar feature based cascade classifiers. Second, to detect dog-images we will use a
pretrained VGG16 model. Finally, after the image is identified as dog/human, we
can pass this image to an CNN model which will process the image and predict the
breed that matches the best out of 133 breeds. </p>

### CNN model created from scratch
<p align="justify">I have built a CNN model from scratch to solve the problem. The model has 3
convolutional layers. All convolutional layers have kernel size of 3 and stride 1. The
first conv layer (conv1) takes the 224*224 input image and the final conv layer
(conv3) produces an output size of 128. ReLU activation function is used here. The
pooling layer of (2,2) is used which will reduce the input size by 2. We have two
fully connected layers that finally produces 133-dimensional output. A dropout of
0.25 is added to avoid over overfitting.</p>

### Refinement - CNN model created with transfer learning
<p align="justify">The CNN created from scratch have accuracy of 13%, Though it meets the
benchmarking, the model can be significantly improved by using transfer learning.
To create <b>CNN with transfer learning</b>, I have selected the <b>Resnet101 architecture</b>
which is pre-trained on ImageNet dataset, the architecture is 101 layers deep. The
last convolutional output of Resnet101 is fed as input to our model. We only need
to add a fully connected layer to produce 133-dimensional output (one for each
dog category). The model performed extremely well when compared to CNN from
scratch. With just 5 epochs, the model got 81% accuracy.</p>

![Sample output](./sample_output.PNG) 

### Model Evaluation
<p align="justify">The CNN model created using transfer learning with
ResNet101 architecture was trained for 5 epochs, and the final model produced an
accuracy of 81% on test data. The model correctly predicted breeds for 680 images out of 836 total images.</p>

**Accuracy on test data: 81% (680/836)**


