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

/**
 * Compresses dynamic range of an image under the assumption that it was equalized
 * @param input CV_8UC1 image
 * @param target_bins number in [1, 255]: number of target bins the modified image will span
 */
void compress_range(Mat input, int target_bins, Mat output) {
    CV_Assert(target_bins >= 1);
    CV_Assert(target_bins <= 255);
    CV_Assert(input.type()==CV_8UC1);

    int table[255];

}
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
