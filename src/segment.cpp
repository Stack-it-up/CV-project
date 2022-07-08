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
    string photos_dir = "../res/evaluation_data/rgb/*.jpg";
    string bboxes_dir = "../res/evaluation_data/det/*.txt";

    vector<string> photos_paths;
    vector<string> bbox_paths;
    glob(photos_dir, photos_paths);
    glob(bboxes_dir, bbox_paths);

    sort(photos_paths.begin(), photos_paths.end());
    sort(bbox_paths.begin(), bbox_paths.end());

    for(int i=20; i<photos_paths.size(); i++) {
        Mat input = imread(photos_paths[i]);
        Mat hsv{};
        cvtColor(input, hsv, COLOR_BGR2HSV_FULL);

        bilateralFilter(hsv.clone(), hsv, 5, 20, 12);

        Mat channels[3];
        split(hsv, channels);
        equalizeHist(channels[0], channels[0]);
        //equalizeHist(channels[1], channels[1]);
        //equalizeHist(channels[2], channels[2]);

        merge(channels, 3, hsv);


        pyrMeanShiftFiltering(hsv, hsv, 3, 10, 5);
        imshow("", hsv);
        waitKey(0);

        vector<Rect> boxes = extract_bboxes(bbox_paths[i]);

        vector<Mat> masks(boxes.size());
        for(int j=0; j<boxes.size(); j++) {
            Rect r = boxes[j];
            Mat bgmodel{};
            Mat fgmodel{};
            grabCut(hsv,
                    masks[j],
                    r,
                    bgmodel,
                    fgmodel,
                    17,
                    GC_INIT_WITH_RECT
            );
        }

        for(Mat mask : masks) {
            Mat output{};
            drawGrabcutMask(input, mask, output, 0.5);
            imshow("", output);
            waitKey(0);
        }

    }
}
