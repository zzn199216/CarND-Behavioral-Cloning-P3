#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examp_pic/center_1.jpg "Center"
[image3]: ./examp_pic/lft.jpg "left Image"
[image4]: ./examp_pic/ctn.jpg "center Image"
[image5]: ./examp_pic/right.jpg.png "right Image"
[image6]: ./examp_pic/raw_pic.png "Normal Image"
[image7]: ./examp_pic/flip_pic.png "Flipped Image"
[image8]: ./examp_pic/cropped.png "Cropped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* model.ipynd notebook with code from model.py and visualization of the model
* run1.mp4 a video recording of the vehicle driving autonomously 2 lap around the track.

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of 3 convolution neural network with 5x5 filter sizes and 2 convolution layers with 3x3 filters after is a flatten and 4 fully-connected layer  .(model.py lines 80-90) 

The model includes RELU layers to introduce nonlinearity (code line 81~85), and the data is normalized in the model using a Keras lambda layer (code line 78). 

####2. Attempts to reduce overfitting in the model

At first I use dropout layers in order to reduce overfitting but later find it be worse.So I no longer use dropout layers in the model. 

The model was trained and validated on different data collected from track 2 and was tested by running it through the simulator on track 1 to avoid overfitting.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 93).

####4. Appropriate training data

Training data was collected collected from track 2 and I make flip for every image  and for the secound and third image , I use correction to make it some noise for reduce overfitting .

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to build a model make car stays on the road.

My first step was to use a convolution neural network model just as what showed in class video 14. I thought this model might be appropriate because it seem to run well in the class video.

But when To auto drive in unity3d car. it run bad on bridge. So I modified the model by make a generator . and make some process on data like add flip image,like make image a little correction as noise, to make it run better

Then I add a dropout layer after the first convolution2D layer. but found it run worse. it always went out the road.So I threw it away~

and after I change some Dense layer and use different data like collected from track1,track2.find out data from track2 run better.might because track2 is more complex.

At the end of the process, the vehicle is able to drive autonomously around the track1 without leaving the road(but not so good on track 2 -,- ).

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with 3 convolution neural network with 5x5 filter sizes and 2 convolution layers with 3x3 filters after is a flatten and 4 fully-connected layer.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| lambda         		| (x / 255.0) - 0.5		 						| 
| Cropping2D         	| 160x320x3 ----->  125x320x3    				| 
| Convolution 5x5     	| 32 5x5 filter, relu activation, subsample 2x2	|
| Convolution 5x5     	| 32 5x5 filter, relu activation, subsample 2x2	|
| Convolution 5x5     	| 32 5x5 filter, relu activation, subsample 2x2	|
| Convolution 5x5     	| 64 3x3 filter, relu activation				|
| Convolution 5x5     	| 64 3x3 filter, relu activation				|
| Flatten				|									 			|
| Dense					|	100	units						 			|
| Dense					|	50	units									|
| Dense					|	10	units									|
| Dense					|	1	units									|


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center , just change the measurement:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

and then cropped it in model:
![alt text][image8]


I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by val_loss increase. I used an adam optimizer so that manually training the learning rate wasn't necessary.
