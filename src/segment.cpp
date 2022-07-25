//
// Created by filippo on 07/07/22.
//
#include  <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include "Util.h"
#include "callbacks.h"
#include "snakes.h"

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
        vector<Rect> boxes = extract_bboxes(bbox_paths[i], 10);
        const char* window_name = "win";
        namedWindow(window_name);

        //// Initial segmentation with meanshift in HSV colorspace //////
        Mat hsv{};
        cvtColor(input, hsv, COLOR_BGR2HSV_FULL);

        pyrMeanShiftFiltering(hsv, hsv, 20, 17.5, 2);
        imshow("", hsv);
        waitKey();

        //////////////// Mask intialization with snakes //////////////////
        vector<Mat> masks(boxes.size()); //contains all bounding boxes detected in the image
        Mat input_gray;
        cvtColor(input, input_gray, COLOR_BGR2GRAY);

        //compute the external force using vector field convolution
        Mat vfc_x, vfc_y;
        int ker_size = max(input.rows, input.cols);
        ker_size /= 2;
        if (ker_size % 2 == 0)
            ker_size -= 1;
        VFC(input_gray, vfc_x, vfc_y, ker_size, 2.4);

        for(int j=0; j<boxes.size(); j++) {
            masks[j] = Mat{input.size(), CV_8UC1, GC_BGD}; //initialize the mask as "definitely background"
            vector<Point> contour = contour_from_rect(boxes[j]);
            fillConvexPoly(masks[j], contour, GC_PR_BGD); //initialize the rectangle as "probably background"

            //DEBUG
            Mat tmp;
            drawGrabcutMask(input, masks[j], tmp, 0.5);
            imshow(window_name, tmp);
            waitKey();
            //TODO remove debug statement

            //compute the snake from the rectangle
            compute_snake(contour, vfc_x, vfc_y, 1, 0.5, 10, 500);
            fillPoly(masks[j], contour, GC_PR_FGD);

            //DEBUG
            drawGrabcutMask(input, masks[j], tmp, 0.5);
            imshow(window_name, tmp);
            waitKey();
            //TODO remove debug statement
        }

        // Possibly TODO adaptive threshold selection by measuring the area variation of the snake


        /*
        ////////// Segmentation with grabcut (mask-initialized) /////////
        vector<Mat> bgm(boxes.size());
        vector<Mat> fgm(boxes.size());
        for(int j=0; j<boxes.size(); j++) {
            Rect r = boxes[j];
            grabCut(hsv,
                    masks[j],
                    r,
                    bgm[j],
                    fgm[j],
                    5,
                    GC_INIT_WITH_RECT
            );
        }

        for(Mat mask : masks) {
            Mat output{};
            drawGrabcutMask(input, mask, output, 0.5);
            imshow("", output);
            waitKey(0);
        }
        */
    }
}
