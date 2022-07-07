Computer Vision Course 2021-2022

Benchmark dataset for Course Project on Human hands detection and segmentation

The benchmark dataset contains a total of 30 images categorized by level of difficulty: 
a) Images with similar backgrounds where few hands are present and clearly visible; 
b) Images with different backgrounds and many hands present, possibly with partial occlusions; 
c) General hand pictures of people of different skin tone and gender. 

Subset a) includes images from 01 to 10
Subset b) includes images from 11 to 20
Subset c) includes images from 21 to 30


All the images are provided in the "rgb" folder.
For each image in the benchmark dataset, the following annotations are provided:

- bounding boxes, in the "det" folder, provided as a text file where each row corresponds to a hand bounding box in the format [x y width height], where x and y mark the top left corner of the bounding box

- segmentation mask, in the "mask" folder, provided as a black and white images with pixel values 0 for the background or 1 for the hands




