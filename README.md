
[car]: ./examples/car.png
[not-car]: ./examples/notcar.png
[car-hog]: ./examples/car_YCrCb_hog.png
[notcar-hog]: ./examples/notcar_YCrCb_hog.png
[s_w_1]: ./examples/sliding_window_1.png
[s_w_2]: ./examples/sliding_window_1_4.png
[s_w_3]: ./examples/sliding_window_1_8.png
[s_w_4]: ./examples/sliding_window_2_2.png

[bbbox1]: ./test_images_output/test1.jpg
[bbbox2]: ./test_images_output/test2.jpg
[bbbox3]: ./test_images_output/test3.jpg
[bbbox4]: ./test_images_output/test4.jpg
[hm1]: ./examples/hm1.png
[hm2]: ./examples/hm2.png
[hm3]: ./examples/hm3.png
[hm4]: ./examples/hm4.png
[label1]: ./test_images_output/heatmap_test1.jpg
[label2]: ./test_images_output/heatmap_test2.jpg
[label3]: ./test_images_output/heatmap_test3.jpg
[label4]: ./test_images_output/heatmap_test4.jpg
[final]: ./test_images_output/label_test1.jpg
[video1]: ./project_video.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. Extracted HOG features from the training images.

The code for this step is contained in lines 18 through 27 of the file called `svc.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car]
![notcar][not-car]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed some images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][car-hog]
![alt text][notcar-hog]

#### 2.Selection of HOG parameters.

I used parameters from course example code: 9 orientations, 8 pixels per cell, 2 cells per block. And it has more than 93 percent accuracy, so I don't have much motivation to try new parameters.

#### 3. Train SVM.

I trained a linear SVM using 80% of all GTI and KITTI vehicle and non-vehicle images with default hyper parameters.

### Sliding Window Search

#### 1. Implementation

I implemented the sliding window from line 25 to line 110 in pipeline.py. Basically it search from y == 400 to y == 650. This range covers all the road in most of the frames.

Windows moves with 75% overlap each step. From the virtualization of searching windows, It already has enough resolution to generate several boxes for a car and will not have too much burdens for process.

It sample each frame with image's width and height from 1 to 1/2.4 and increase dominator scales 0.2 per step. I tried different scales. Window scales larger than 2.5 is to large for the image. Other cars shouldn't that close to camera. Also too many windows sizes and scales will affect performance. I don't want to wait for one hour to process the project video.

![alt text][s_w_1]
![alt text][s_w_2]
![alt text][s_w_3]
![alt text][s_w_4]

#### 2. Examples

Ultimately I searched on eight scales using YCrCb 3-channel HOG features which provided a nice result. Here are some example images:

![alt text][bbbox1]
![alt text][bbbox2]
![alt text][bbbox3]
![alt text][bbbox4]
---

### Video Implementation

#### 1. Video Link
Here's a [link to my video result](./output_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Four test images and their corresponding heatmaps:

![alt text][hm1]
![alt text][hm2]
![alt text][hm3]
![alt text][hm4]


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from four test images:
![alt text][label1]
![alt text][label2]
![alt text][label3]
![alt text][label4]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][final]



---

### Discussion

#### 1. Performance Issue
hog method from skimage is largely implemented by python. So it's performance is terrible. On my laptop, handle one frame cost around 1.5 seconds. So this whole code isn't possible to run on a real car.

#### 2. Partial Cars and Closed Cars.
My code has difficulties to identified partial cars on image. Maybe more samples contains partial cars would be helpful. Also, it will combine multiple box together of two or more cars are to close to each other.


#### 3. CNN + Sliding Window?
Through the whole term 1, we just tried CNN to classify traffic signs. But I companies working on self-driving cars actually using trained CNN to identify different objects on road?
