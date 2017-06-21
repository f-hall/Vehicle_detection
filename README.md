[//]: # (Image References)
[image1]: ./output_images/car.png
[image2]: ./output_images/Ychannel.png
[image3]: ./output_images/size1.png
[image4]: ./output_images/original_1.jpg
[image5]: ./output_images/original_2.jpg
[image6]: ./output_images/test_1.png
[image7]: ./output_images/row1.png
[image8]: ./output_images/label.png
[video1]: ./output.mp4

##Writeup for Project 5 - Vehicle Detection

###General Information
You can find the commented code for this Project in the notebook `P5.ipynb` or in the Python-file `P5.py`. All images in this writeup can be found in the directory `output_images`. Additional test-images can be found the folders `v1` and `v2`. The result video `output.mp4`  is located in the main folder.


###Training- and Testset

The data I used to train the classifier comes from the GTI and KITTI folders provided by udacity - both have labeled (through different folders) sample-data for vehicles and non-vehicles. It is necessary to mention that the vehicle-images from the GTI-folders are time-series data and to avoid the appearance of nearly identical images in the training- and the testset I created the function `selecting_sets`. With this function one can select random consecutive image-chunks of a specific count (I choose 100) to be chosen for the testset, so that we can minimize this issue. Another approach would be to just take the first 20% from the GTI-vehicle-folder, but that would not randomize the data enough from my point of view. 

I choose a 80% - 20% split for the training- and the testset.

###Histogram of Oriented Gradients (HOG)

I used the functions `get_hog_features` and `extract_features`, which are using the `skimage-hog()`-function, from the udacity-lessons to extract the HOG-features from both the vehicle-images and the non-vehicle-data. I tried different parameters and came to the conclusion that the `YCrCb` color space, `orient=12`, `pixel_per_cell=8` and `cell_per_block=2` are working really well for me. Furthermore I choose to select all three color-channels from the `YCrCb` color space to gather enough information. It should be said that `YUV` and `HLS` had just a slightly worse performance that the selected and furthermore the pick of `orient=12` might be an overkill, where numbers of 8 or 9 may be enough.

Here is an example using the given parameters for a vehicle and a non-vehicle:
![alt text][image1]

Below are the extracted HOG-features for both images and all three channels:

![alt text][image2]

I used the `StandardScaler().fit()`-function and then `X_scaler.transform()` to normalize the feature-vectors. It should be mentioned that I used only the trainings-data to fit the Scaler and after that used it to transform both the training- and the test-features. It should be used this way, because we only use the training-set to train our Classifier for whom the data should be normalized.

###Classifier

I trained a SVM, herefore I used the function `svm.SVC()`, with the standard parameters - the kernel as 'rbf' and the parameter 'C=1.0', which has given me really good results. The test-accuracy is somewhere between 98% and 99.5%. I tried some other classifier like the `Random-Forest` and `k-Nearest-Neighbors`-approaches with different parameters. I got also good outputs with them but not quite as stable and outstanding as with the SVM.

###Sliding Window Search

I used the Sliding Window Search `find_cars` from the udacity-lessons (slightly altered) to search in an image for possible vehicles. The only parameters still open to select for this function were the window for where to search - `ystart` and `ystop` - and the sizes of the search-patches, for which I used the variable `scale` (size = standardsize(64x64pixels) * scale).

I created the function `find_cars_range` to use the sliding window search for different `scales`=[1, 1.5, 1.75, 2, 2.5, 3]. Furthermore I used `ystart`=380 and `ystop`=656, altered for scale under 1.8 with `ystop`=530 and lower than 1 with `ystop`=500. With that we can save computational time, because we can assume that vehicle-images with a small scale are not found at the bottom of an image, which is quite important, because I took an approach, where I let the search-patches overlap generously to get a good result at the expense of a fast computational time - here could be changes made, because it will take a really long time (hours...) without the help of a gpu to get the result for the 50 seconds video. Here are the search-patches for the different `scales`:

![alt text][image3]

Here you can see two testimages as well as their output from `find_cars_range`:

![alt text][image4]

![alt text][image5]

![alt text][image6]

As explained before, without a gpu it will take a really long time to search in every patch with help of our SVM, especially if it isn't just an image, but a video with over 1000 frames. To speed things at least a little bit up I wrote a function called `help_function()` which uses parallelization for all 4 cpu-kernels for when we have to search in a video for vehicles.

I pickled the list of the found vehicle-rectangles in the data `list.pkl`.

###Heatmap and Labels

We can reduce false positives in the video by weighting in some picture-results from before and after the actual frame (just before when using on a stream I suppose) - I'm using two frames before and two frames after the actual image. To display this idea you can see five following frames from the folder `v2` and their associated heatmaps:

![alt text][image7]

With the additional information I used a threshold for the heatmaps and then the function `scipy.ndimage.measurements.label()` to get the boundary box and therefore the output each frame and the video:

![alt text][image8]

### Video Implementation

Here's a [link to my video result](./output.mp4)

###Discussion

The first thing to approach would be to fasten things up, maybe using a gpu should enough, but maybe one can tweak my implementation here and there to speed things a bit up.

Furthermore this is a project where one could experiment a lot with different parameters and variables, like selection of classifiers and features. The SVM seems to be really good, but maybe something better (or faster but nearly as good) can be used. For the future one could face the challenge to measure how far away or even how fast the found vehicles approximately would be. There are a lot more things to explore in this project.

Thanks in advance for the review :)

