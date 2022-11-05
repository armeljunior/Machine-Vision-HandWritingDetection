# MachineVision

Implementation and evaluation of a classification model for handwritten numbers . Two classifiers were explored, kNN and AlexNet (deep learning), the effectiveness of each was analyzed and compared against one another. The accuracy of AlexNet substantially outperformed kNN because the neural network approach was more precise in recognizing patterns within the dataset. Once the models were assessed, the process of improving and optimizing the algorithm could begin. To improve the algorithm, the training and testing process was analyzed to increase the performance and to explore the effects of trying to identify other characters (A-Z) and handwriting styles. However, this caused the accuracy to drop from 97% to 88% due to misidentification by the classifier. The algorithm was also optimized by using the k-fold method to validate the accuracy by testing against every image over k iterations, therefore, having a better representation of how the model performs. 
## **kNN Classification:**


This section aims to use the Hu moments (Ming-Kuei Hu, 1962) of the numeral images to train a kNN classifier which can then be applied to unlabelled images to generate a classification. The process applied for this section is split into the three main parts of building and training the classifier, calculating the Hu moments of the individual images, and evaluating the accuracy of the classifier.

<img width="270" alt="image" src="https://user-images.githubusercontent.com/90009399/200141442-75768246-dac4-4002-b39d-e550ecfc68c8.png">

## Numerical Extraction:

1.Taking the image input of a string of characters, the image is then converted from its original colour space to a binary format.

2.The converted image is then analyzed to determine the boundaries and number of characters by using their white pixel perimeters as indicators of their locations.

3.Using the known boundaries and number of characters, the converted image is then segmented into sub-images for individual classification.

4.Each sub-image is then padded with black pixels to fit the dimension requirements for inputs into each algorithm.

5.As each sub-image is passed to a classifier, the outputs are then used to reconstruct the string value found in the original image.


<img src="https://user-images.githubusercontent.com/90009399/200142101-febff13d-1d89-4366-8cac-09d326e0960f.png" width="100" height="100" /> <img src="https://user-images.githubusercontent.com/90009399/200142104-84d9b87f-01e7-4814-b0ff-6fe59d8c4a69.png" width="100" height="100" /> <img src="https://user-images.githubusercontent.com/90009399/200142105-150b7336-a5d9-4780-ad2d-aec4c97d71bb.png" width="100" height="100" /> <img src="https://user-images.githubusercontent.com/90009399/200142106-f8b320e4-1aad-4a8e-be03-2e549d73b0b3.png" width="100" height="100" /> <img src="https://user-images.githubusercontent.com/90009399/200142107-1607c290-a247-4df0-b026-87a29d538d6d.png" width="100" height="100" />







## Deep Learning

Using the AlexNet design, a convolutional neural network, to develop an algorithm using numerical images to train a classifier that can accept images of multi-digit numbers and return their values. The process in developing the algorithm is to first divide the provided photos into two sets where the one set will be used to train the neural network. The second set will be used to determine the network's accuracy in classifying untrained images. Within the neural network to be trained there are eight main layers, the first five are used to learn low-level features such as edges and blobs, while the final three are used to learn more specific features.

<img width="600" alt="AAPicture 1" src="https://user-images.githubusercontent.com/90009399/200141945-57c8fa67-b18d-475a-a13d-8f4343cb9d23.png">


### Training and Validation of data

<img width="470" alt="aaaa" src="https://user-images.githubusercontent.com/90009399/200143123-30ef9010-9235-4714-acdf-d2a499c08fe9.png">

### Confusion Matrix
<img width="323" alt="confmatrix" src="https://user-images.githubusercontent.com/90009399/200143124-e85a987f-721e-48f8-8b16-e699ae20b84c.png">












