#**UDACITY Self Driving Car Term1 Project 2: Traffic Sign Recognition by Berk Tepebağ** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale2.jpg "Grayscaling"
#[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./images/1-22.jpg "Traffic Sign 1"
[image5]: ./images/2-17.jpg "Traffic Sign 2"
[image6]: ./images/3-4.jpg "Traffic Sign 3"
[image7]: ./images/4-20.jpg "Traffic Sign 4"
[image8]: ./images/5-40.jpg "Traffic Sign 5"
[image9]: ./images/6-27.jpg "Traffic Sign 6"
[image10]: ./images/7-28.jpg "Traffic Sign 7"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.
##TODO: Delete after check
You can check my project from github -> [project code](https://github.com/berktepebag/carnd-term1-project2-traffic-sign-classifier/blob/master/
Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32x3.
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because colors causes an increase in need of computational power and do not worth using colored images since signs are obvious even if they are gray.  

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
#For more info on this:
#https://stats.stackexchange.com/questions/185853/why-do-we-need-to-normalize-the-images-before-we-put-them-into-cnn

As a last step, I normalized the image data because not normalizing may cause correcting one weight more than another weight. Backpropagation  depends on learning rate and change of weights so if we do not normalize it will cause oscillation in costs which will cause lower accuracy.

##Will do extra part later.
I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:
##
![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:
#Followed the instructions from: 
#http://cs231n.github.io/convolutional-networks/#conv

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 24x24x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x16 				|
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 11x11x32	|
| RELU					|												|
| Convolution 5x5    	| 2x2 stride, valid padding, outputs 5x5x64		|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 4x4x64	 				|
| Fully connected		| Input = 1024. Output = 512.      				|
| Fully connected		| Input = 512. Output = 256.      				|
| Fully connected		| Input = 256. Output = 128.      				|
| Fully connected		| Input = 128. Output = 43.      				|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer with 100 Epochs and 128 batch. Learning rate was 0.0001, tried 0.00001 but it was learining too slow and with 100 epochs it was under 90% accuracy. So 0.0001 was a better choice with 93% accuracy. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ..
* validation set accuracy of 93.1%
* test set accuracy of 91.8%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I used the same architecture of LeNet but accuracy was not going above 90% so I followed the Stanfords cs231n lecture (http://cs231n.github.io/convolutional-networks/#conv) and took 2x(2 conv 1 maxpooling) and 3 fully connected layers.

* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

(LeNet's) Max pooling after each Conv cause too deep layers too early which caused low accuracy. It is easier to add more conv layers with this method (2 conv 1 maxpool) before getting too deep. This helped increasing the accuracy.

* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

***Pre-run Predictions For the Internet Images:

1. Image: "Image is looking straight to camera, brightness and contrast is not so well, image is bit grayish which may cause problems when applying gray scale.."
2. Image: "Image is looking to camera straight forward, brightness and contrast is almost perfect. Classification should be done with out problem."
3. Image: "Image is taken from left bottom side of the sign, disturbance in alignment may cause wrong classification of the sign."
4. Image: "Image is taken from right bottom side of the sign, disturbed alignment may cause wrong classification of the sign. Also image is bit grayish which can cause problem when applying gray scale."
5. Image: "Image is taken from left bottom side of the sign, disturbance in alignment may cause wrong classification of the sign. Birghtness and contrast is not a problem."
6. Image: "Image is taken almost straight, brightness and contrast is good for classification. "
7. Image: "Image is taken almost straight, brightness and contrast is fairly good for classification."]

Here are the results of the prediction:

| Image			       	 		|     Prediction	        					| 
|:-----------------------------:|:---------------------------------------------:| 
| Bumpy Road	   				| Bumpy Road   									| 
| No Entry 						| No Entry										|
| Speed limit (70km/h)			| Speed limit (30km/h)							|
| Dangerous curve to the right	| Turn right ahead				 				|
| Roundabout mandatory			| Priority road   								|
| Pedestrians					| General caution								|
| Children crossing				| Children crossing								|


The model was able to correctly guess 3 of the 7 traffic signs, which gives an accuracy of 42.9%. This is far from test accuracy but since images were not perfectly aligned or distorted this is acceptable.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located just before the "project writeup" section of the Ipython notebook.

For a good prediction, the model is hundred percent sure that this is a "Bumpy road" sign (probability of 1), and the image does contain a "Bumpy road" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Bumpy Road  									| 
| 1.18827614e-09		| Bicycles crossing								|
| 2.20850858e-13		| Road work										|
| 5.51253243e-14		| Children crossing				 				|
| 2.67323069e-15		| Road narrows on the right     				|

For a bad prediction, the model is hundred percent sure that this is a "Priority road" sign (probability of 1), and the image does contain a "Roundabout mandatory" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority road 								| 
| 1.15141834e-16 		| Roundabout mandatory							|
| 9.04314293e-31		| Right-of-way at the next intersection			|
| 8.31188764e-31		| Children crossing				 				|
| 5.52404859e-36		| Keep Right-of-way    							|

Second prediction is true but probabilty is so low that we can expect it as a 100% wrong prediction.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


