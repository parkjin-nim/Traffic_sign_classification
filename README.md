# **Traffic Sign Recognition** 

## Writeup

---

**The goals / steps of this project are the following:**
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/original.png "Visualization"
[image2]: ./writeup_imgs/dataset_dist.png "Dataset distribution"
[image3]: ./writeup_imgs/equal_hist.png "Histogram equalization"
[image4]: ./writeup_imgs/image_generate.png "Generated images"
[image5]: ./writeup_imgs/image_normalize.png "Normalized distribution"
[image6]: ./five_test_signs/30kph.jpeg "Speed limit (30km/h)"
[image7]: ./five_test_signs/keep-right.jpeg "Keep right"
[image8]: ./five_test_signs/stop.jpeg "Stop"
[image9]: ./five_test_signs/roundabout.jpeg "Roundabout mandatory"
[image10]: ./five_test_signs/children-crossing.jpeg "Children crossing"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 208794(augmented by original 34799)
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

First, let's look at how the data set look like. Here is an exploratory visualization of the data set. And the observation is that input images has many issues such as viewpoint variations, lighting conditions (saturations, low-contrast), motion-blur, occlusions, sun glare, physical damage, colors fading, graffiti, stickers and low resolution.
Above all, different lighting conditions seem to be the biggest issue. Some samples are so dark that anyone can hardly recognize. This variation may also be a problem for machines to learn. So, i'm going to deal with this issue later at the pre-processing section.

![alt text][image1]

Here is another exploratory visualization. The bar char is data histogram. By the look of it, it shows that each class of dataset has very different number of samples. This could lead to a natural bias that might actually be happening on the street in the real world. Checking the distribution of test data, test data set has almost similiar distribution with that of train data set.

But, labels such as 0, 19, 24, 27, 29, 32, 37, 41, and 42 are less than 250 samples. This lack of samples for some lables might contribute to the bias. We want the model to be perceivable enough for less frequent signs too, not overfitting. It may be reasonable to fill those cavities to prevent the bias.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, we need to have better quality of images. I'm using [Histogram equalization](https://en.wikipedia.org/wiki/Histogram_equalization) for the first pre-processing. As pointed-out above at the exploratory visualization of the dataset, the different lighting conditions among samples are the primary issue that must be dealt with. In order to do so, i'm trying to convert color space from RGB to [LAB](https://en.wikipedia.org/wiki/CIELAB_color_space). The L channel of LAB represents brightness. By equalizing the histogram distribution of L channel, we could obtain the images of averaged brightness. Color channels need not change, so i leave A,B channels intact. The below is the equalized L-channel signs. 

![alt text][image3]

As next step, training dataset needs to be augmented. How the augmentation is enforced is examplified on the [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It says "Samples are randomly perturbed in position ([-2,2] pixels), in scale ([.9,1.1] ratio) and rotation ([-15,+15] degrees)". To do this transformations, i'm trying to use the Keras api, ImageDataGenerator. And set the translation(-.1~.1), rotation((-15,15) degree), scaling(0.5~1.05), and shear(0.5) as paramter. I'm augmenting the numble samples for each label that has less than 800 samples to make sure that it has  more than 800 samples. Those labels are [0, 6, 14, 15, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39, 40, 41, 42]. The number of total training samples are now 51448.


Here is examples of augmented image:

![alt text][image4]

As a last step, I normalized the image data using (pixel - 128)/ 128 as suggested and shuffled training data set.
Here is an example normalized distribution of an image:

![alt text][image5]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 14x14x12  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x28 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 5x5x28  					|
| Flatten				| output 700									|
| Fully connected		| input 700 outputs 300  				|
| RELU					|												|
| Dropout				| keep prob. 0.5								|
| Fully connected		| input 300 outputs 120  						|
| RELU					|												|
| Dropout				| keep prob. 0.7								|
| Fully connected		| input 120 outputs 43  							|
| Softmax				| etc.        									| 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer, and tested performance with varying learning rate from 0.01,0.001,and 0.0005; EPOCHS from 50 tp 100. Since data augmentation is used, its effect would be worth mentioning. Overall observation is, training with augmented data increases about 1% accuracy performance(give or take).

Here is the example of validation accuracy over EPOCHS 50(learning rate 0.001).
| EPOCHS | Acc. w/ 34799	| Acc. w/ 208794	| 
|:------ |:----------------:|:-----------------:| 
| 10.    | .93				| .94				| 
| 20.    | .94				| .95				|
| 30.    | .94				| .96				|
| 40.    | .95				| .96				|
| 50.    | .95				| .96				|

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 97%
* test set accuracy of 95%

I used Lenet5 architecture as guided. Lenet5 architecture was designed for a grayscale image as input. Now that the input is color images, network should be modified somehow to embrace the enlarged dimesion of images. I modified Lenet5 architecture to capture more low-level features. I simply made feature-maps go deeper such that conv1 output is changed from 28x28x6 to 28x28x12, conv2 from 10x10x16 to 10x10x28, and flatten output from 400 to 700.

I set the learing rate low enough(0.0005) and epochs high enough(100). And now that i have 700 flatten featuremap output, i put 2 dropouts at the classifier to prevent overfitting. Alexnet(2012) architecure could be chosen. The Network had a very similar architecture to LeNet, but was deeper, bigger, and featured Convolutional Layers stacked on top of each other (previously it was common to only have a single CONV layer always immediately followed by a POOL layer).

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			       						| Prediction		        					| 
|:-----------------------------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)         				| Speed limit (30km/h)							| 
| Keep right								| Keep right									|
| Stop										| Stop											|
| Roundabout mandatory						| Roundabout mandatory				 			|
| Children crossing							| Children crossing 							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For all 5 images, the highest scoreo of each label has a very large margin over the 2nd highest. The model has high confidence(almost 1.0) for correct labels. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.					| Speed limit (30km/h) 							| 
| 1.					| Keep right									|
| 1.					| Stop											|
| 1.					| Roundabout mandatory				 			|
| 1.					| Children crossing								|
