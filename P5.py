
# coding: utf-8

# In[ ]:

# importing all necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')
import glob
from moviepy.editor import VideoFileClip
import math
import random
from PIL import Image
import pickle
import pylab
import imageio
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label

import multiprocessing
from multiprocessing import Process

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[ ]:

# Setting procentage of the data used for the testset (20% here)
# The parameter follow is the size of the image-chunks we are getting from the gti-vehicles folder to solve the problem
# with the rowing of the data. (For example: the pictures 42-47 are really similar, so we want them in the same set (test or training))
# Another easier solution would be just taking 20% in a row from the folder, but that is not random enough for my liking.
test_set_percent = 20
follow = 100


# In[ ]:

# Getting pathes for vehicle-images
veh_kitti = sorted(glob.glob('/media/frank/Zusatz/CarND-Vehicle-Detection-master/vehicles/KITTI_extracted/*.png'))
veh_gtifar = sorted(glob.glob('/media/frank/Zusatz/CarND-Vehicle-Detection-master/vehicles/GTI_Far/*.png'))
veh_gticlose = sorted(glob.glob('/media/frank/Zusatz/CarND-Vehicle-Detection-master/vehicles/GTI_MiddleClose/*.png'))
veh_gtiright = sorted(glob.glob('/media/frank/Zusatz/CarND-Vehicle-Detection-master/vehicles/GTI_Right/*.png'))
veh_gtileft = sorted(glob.glob('/media/frank/Zusatz/CarND-Vehicle-Detection-master/vehicles/GTI_Left/*.png'))


# In[ ]:

# Getting pathes for non-vehicle-images
non_extras = sorted(glob.glob('/media/frank/Zusatz/CarND-Vehicle-Detection-master/non-vehicles/Extras/*.png'))
non_gti = sorted(glob.glob('/media/frank/Zusatz/CarND-Vehicle-Detection-master/non-vehicles/GTI/*.png'))


# In[ ]:

# function for selecting test-sets with the follow-parameter (only necessary for gti-vehicles)
def selecting_sets(vehicle_list = [], follow=10, test_set_procent=10):
    jo = int(len(vehicle_list)*(test_set_procent/100))
    random_list = []
    test_list = []
    train_list = []
    while len(random_list) < jo:
        test = random.randint(0, len(vehicle_list)-1-follow)
        while test in random_list:
            test = random.randint(0, len(vehicle_list)-1-follow)
        for i in range(follow):
            if test+i not in random_list:
                random_list.append(test+i)
            
    while len(random_list) > jo:
        random_list.pop()
        
    for i in range(len(vehicle_list)):
        if i in random_list:
            test_list.append(vehicle_list[i])
        else:
            train_list.append(vehicle_list[i])
            
    random.shuffle(train_list)
    random.shuffle(test_list)
    return(train_list, test_list)
        


# In[ ]:

# Splitting all folder-sets in training and testdata (vehicles)
train_far, test_far = selecting_sets(veh_gtifar, follow, test_set_percent)
train_close, test_close = selecting_sets(veh_gticlose, follow, test_set_percent)
train_right, test_right = selecting_sets(veh_gtiright, follow, test_set_percent)
train_left, test_left = selecting_sets(veh_gtileft, follow, test_set_percent)
kitti_help = int(len(veh_kitti)*(test_set_percent/100))
random.shuffle(veh_kitti)
train_kitti, test_kitti = veh_kitti[kitti_help:], veh_kitti[:kitti_help]


# In[ ]:

# Splitting all folder-sets in training and testdata (non-vehicles)
gti_help = int(len(non_gti)*(test_set_percent/100))
random.shuffle(non_gti)
train_gti, test_gti = non_gti[gti_help:], non_gti[:gti_help]

extras_help = int(len(non_extras)*(test_set_percent/100))
random.shuffle(non_extras)
train_extras, test_extras = non_extras[extras_help:], non_extras[:extras_help]


# In[ ]:

# Fusion the training and testdata to one set each and shuffling them
test_vehicle = test_far + test_close + test_right + test_left + test_kitti
train_vehicle = train_far + train_close + train_right + train_left + train_kitti
test_non = test_gti + test_extras
train_non = train_gti + train_extras

random.shuffle(test_vehicle)
random.shuffle(train_vehicle)
random.shuffle(test_non)
random.shuffle(train_non)


# In[ ]:

# function to get hog features of an image with choosen orient, pix per cell and cell per block
# (From udacity-lessons)
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
# function to extract hog features from an image
# (From udacity-lessons)
def extract_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    features = []
    for file in imgs:
        image = mpimg.imread(file)
        if file.endswith('.png'):
            image = cv2.convertScaleAbs(image*255)
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        features.append(hog_features)
    return features
    
    


# In[ ]:

# Choose variables for hog-extraction
# This involves a lot of trial and error, but I came to the conclusion that this parameters
# works best for me.
colorspace = 'YCrCb' # Colorspace HLS/YUV were nearly as good
orient = 12 # Tested also with 9, but I think 12 is a bit more stable
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Did not test a lot with just 1 Channel. We should take all information we can get here, I suppose.

# Extract hog-features for the vehicle-data
veh_features = extract_features(train_vehicle, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)

# Extract hog-features for the non-vehicle-data
non_features = extract_features(train_non, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)


# In[ ]:

# Getting the labels ready for training data
y = np.hstack((np.ones(len(veh_features)), 
              np.zeros(len(non_features))))


# In[ ]:

# Getting feature vector ready for training data
x = np.asarray(veh_features + non_features).astype(np.float64)


# In[ ]:

# Normalizing the training data
# It's important in theorie that you normalize using just the training data (not the testset) unlike in the udacity-lessons.
# But in the end it should not matter much, I suppose.
X_scaler = StandardScaler().fit(x)
scaled_x = X_scaler.transform(x)


# In[ ]:

# Use a SVM to get a classifier (using our feature-vector scaled_x and label-vector y)
# The standard parameters for svm.SVC seems really good (tried a bit with other kernels and Cs)
clf1 = svm.SVC()
clf1.fit(scaled_x, y)


# In[ ]:

# Getting everything ready for the testset like before with the trainingset
veh_features_test = extract_features(test_vehicle, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)

non_features_test = extract_features(test_non, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)

y_test = np.hstack((np.ones(len(veh_features_test)), 
              np.zeros(len(non_features_test))))

x_test = np.asarray(veh_features_test + non_features_test).astype(np.float64)

scaled_x_test = X_scaler.transform(x_test)


# In[ ]:

# Checking the test accuracy of our Classifier (Something around 98.5% - 99.5% most of the time)
print('Test Accuracy of SVC = ', round(clf1.score(scaled_x_test, y_test), 4))


# In[ ]:

# Function to find cars on an image using hog-features and the classifier
# (Mostly taken from udacity-lessons. I commented of changed lines)
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, steps):
    voidlist = []
    draw_img = np.zeros_like(img)
    
    # For small scales I use a smaller search window, as small vehicles will not be near the bottom of the image
    if scale <= 1.80:
        ystop = 530
    
    if scale <= 1:
        ystop = 500
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    # I choose the steps in x-direction fixed as 1. For testing purposes I used the parameter steps for the y-direction.
    # Lastly I came to the conclusion that steps=1 for all scales is the most reliable choice albeit needing longer to compute. 
    cells_per_stepx = 1
    cells_per_stepy = steps  # Instead of overlap, define how many cells to step
    if scale >= 2:
        cells_per_step1 = 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step1
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            test_features = X_scaler.transform(hog_features).ravel()
            test_prediction = svc.predict(test_features)
             
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale) 
                help_here = np.array([xbox_left, ytop_draw+ystart, xbox_left+win_draw, ytop_draw+win_draw+ystart])
                voidlist.append(help_here)
                
    # Returning all rectangles, where vehicles where found in a list.            
    return voidlist


# In[ ]:

# General pipeline using find_cars for differet sizes and steps, saving the output in one new list.
def find_cars_range(image):
    patch_list = []
    for i in range(6):
        size = [1, 1.5, 1.75, 2, 2.5, 3]
        steps = [1, 1, 1, 1, 1, 1]
        patch_list.extend(find_cars(image, ystart, ystop, size[i], clf1, X_scaler, orient, pix_per_cell, cell_per_block, steps[i]))#, spatial_size, hist_bins)
    for i in patch_list:
        cv2.rectangle(image,(i[0], i[1]),(i[2],i[3]),(0,0,255),6)
    # Return image with positiv-patches drawn on it and the list with all positiv-patches
    return image, patch_list


# In[ ]:

# Choosing parameters used for where to search vehicles in general.
ystart = 380
ystop = 656


# In[ ]:

# Loading the Video and get all frames with the iter_frames function
clip = VideoFileClip('/media/frank/Zusatz/CarND-Vehicle-Detection-master/project_video.mp4')
clip_frames = clip.iter_frames()


# In[ ]:

# Using parallelization to use all 4 cpu-kernels
# This is not necessary, but it takes a really long time (hours) to search in all patches for vehicles without a gpu
# So to get things slightly faster done I took this route.
count = 0

def help_function(clip_frame):
    global count
    print(count)
    print(datetime.datetime.now().time())
    count = count+1
    output_carsearch = find_cars_range(clip_frame)
    return output_carsearch[1]

pool = multiprocessing.Pool(processes=4)    
results = pool.imap(help_function, clip_frames)
pool.close()
pool.join()
patch_list=[]
for result in results:
    patch_list.append(result)


# In[ ]:

# I pickle the list so I don't have to search for cars in the video anew
# This has saved me a lot of time...
with open('list.pkl', 'wb') as f:
    pickle.dump(patch_list, f)


# In[ ]:

with open('list.pkl', 'rb') as f:
    load_list = pickle.load(f)


# In[ ]:

load_list = patch_list


# In[ ]:

# To get rid of false positives I make a new list for the multiple detection suggested from udacity
list1 = list(load_list)
list2 = list(load_list)
list3 = list(load_list)
list4 = list(load_list)
list5 = list(load_list)

list1.insert(0, []), list1.insert(0, []), list1.insert(0, []), list1.insert(0, [])
list2.append([]), list2.insert(0, []), list2.insert(0, []), list2.insert(0, [])
list3.append([]), list3.append([]), list3.insert(0, []), list3.insert(0, [])
list4.append([]), list4.append([]), list4.append([]), list4.insert(0, [])
list5.append([]), list5.append([]), list5.append([]), list5.append([])

whole_liste = []
for i in range(len(list3)):
    whole_liste.append(list1[i]+list2[i]+list3[i]+list4[i]+list5[i])
del whole_liste[0]
del whole_liste[0]
del whole_liste[-1]
# See last cell to see why the next line is outcommented
#del whole_liste[-1]
print(len(whole_liste))


# In[ ]:

# Using heatmaps and labels to get a patch for each car on the frame
# (taken from udacity-lessons and altered)
counter = 0
def video_pipeline(image):
    # There is surely a better way than using global parameters (never a good idea)
    global counter
    global whole_liste
    heat = np.zeros_like(image[:,:,0])
    list_patch = whole_liste[counter]

    counter = counter+1
    if list_patch != []:
        for i in list_patch:
            heat[i[1]:i[3], i[0]:i[2]] += 1
        # Choosing heat threshold
        heat[heat <= 11] = 0
        heat = np.clip(heat, 0, 1)
        labels = label(heat)
        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            cv2.rectangle(image,bbox[0],bbox[1],(0,0,255),6)
    return image


# In[ ]:

# Using the function on the project_video.mp4 and making a new video output.mp4
output_video = 'output.mp4'
clip = VideoFileClip('/media/frank/Zusatz/CarND-Vehicle-Detection-master/project_video.mp4')

# For some reason the function write_videofile do not take the framerate (25) from the clip in, but
# just use the standard (24). So I have to set the parameter myself to 25, but now he gives me 1261 frames
# instead of 1260 (length of whole_list). Therefore I commented del whole_list out some cells above. 
# Not the best workaround... Have to find out why write_videofile does that so badly - maybe I will switch to pure ffmpeg
# or opencv-videowriting in the future, but for now it have to do.
output_clip = clip.fl_image(video_pipeline)
get_ipython().magic('time output_clip.write_videofile(output_video, audio=False, fps=25)')

HTML("""
<video width="1280" height="720" controls>
   <source src="{0}">
<\video>
""".format(output_video))


# In[ ]:



