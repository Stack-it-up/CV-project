//
// Created by filippo on 07/07/22.
//
#include  <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include "Util.h"

using namespace cv;
using namespace std;

int main() {
    string photos_dir = "/home/filippo/Desktop/unipd/4anno/computer_vision/final_project/datasets/Dataset progetto CV - Hand detection _ segmentation/rgb/*.jpg";
    string bboxes_dir = "/home/filippo/Desktop/unipd/4anno/computer_vision/final_project/datasets/Dataset progetto CV - Hand detection _ segmentation/det/*.txt";

    vector<string> photos_paths;
    vector<string> bbox_paths;
    glob(photos_dir, photos_paths);
    glob(bboxes_dir, bbox_paths);

    sort(photos_paths.begin(), photos_paths.end());
    sort(bbox_paths.begin(), bbox_paths.end());

    for(int i=0; i<photos_paths.size(); i++) {
        Mat input = imread(photos_paths[i]);
        Mat hsv{};
        cvtColor(input, hsv, COLOR_BGR2HSV_FULL);

        Mat channels[3];
        split(hsv, channels);
        channels[2] = 100;
        channels[3] = 100;
        merge(channels, 3, hsv);


        pyrMeanShiftFiltering(hsv, hsv, 3, 10);
        imshow("", hsv);
        waitKey(0);

        //vector<Rect> boxes = extract_bboxes(bbox_paths[i]);
    }
}
