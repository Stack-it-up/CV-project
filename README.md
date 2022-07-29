# Hand Detect
*final project for the Computer Vision course @UniPD. A.Y. 2021-2022*

##Compilation on Linux
```
   mkdir build
   cd build
   cmake ..
   make 
```
##Run the demo
```
./hand_detect
```
runs the demo on the whole test set, showing the results. Please remember to press a key after the final
detection or segmentation result is shown! by default, the segmentation process
also shows the intermediate steps (meanshift, bounding boxes and snakes).

##Just one image
To run the whole process on one single image:

```
./hand_detect -d 1 -s 1 -image path_to_image -bb path_to_bound_box 
```
##Detailed commands
* ```-d 0``` Detect and don't show the resulting image
* ```-d 1``` Detect and show the resulting image
* ```-s 0``` Segment and don't show the intermediate steps
* ```-s 1``` Segment and show the intermediate steps
* ```-image path``` Specify the path to the image you want to process. If this parameter
is not provided, the computation will be on the whole test set.
* ```-bb path``` Specify the folder in which the bounding boxes file will be written
* ```  -h ``` Show a help message and exit
