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

    for(int i=5; i<photos_paths.size(); i++) {
        Mat input = imread(photos_paths[i]);
        vector<Rect> boxes = extract_bboxes(bbox_paths[i], 10);
        const char* window_name = "win";
        namedWindow(window_name, WINDOW_NORMAL);

        //// Initial segmentation with meanshift in HSV colorspace //////
        Mat hsv{};
        cvtColor(input, hsv, COLOR_BGR2HSV_FULL);

        pyrMeanShiftFiltering(hsv, hsv, 20, 17.5, 2);
        imshow(window_name, hsv);
        waitKey(0);

        //////////////// Mask intialization with snakes //////////////////
        vector<Mat> masks_snake(boxes.size()); //contains all bounding boxes detected in the image
        vector<Mat> masks_gc(boxes.size());
        Mat input_gray;
        cvtColor(input, input_gray, COLOR_BGR2GRAY);

        //compute the external force using vector field convolution
        Mat vfc_x, vfc_y;
        int ker_size = max(input.rows, input.cols);
        ker_size /= 2;
        if (ker_size % 2 == 0)
            ker_size -= 1;
        VFC(input_gray, vfc_x, vfc_y, ker_size, 2.4);
        imshow(window_name, abs(vfc_x));
        waitKey();

        for(int j=0; j<boxes.size(); j++) {
            masks_snake[j] = Mat{input.size(), CV_8UC1, GC_BGD}; //initialize the mask as "definitely background"
            vector<Point> contour = contour_from_rect(boxes[j]);
            fillConvexPoly(masks_snake[j], contour, GC_PR_BGD); //initialize the rectangle as "probably background"

            //DEBUG
            Mat tmp;
            drawGrabcutMask(input, masks_snake[j], tmp, 0.5);
            imshow(window_name, tmp);
            waitKey(2000);
            //TODO remove debug statement

            //compute the snake from the rectangle
            compute_snake(contour, vfc_x, vfc_y, 0.3, 0.5, 2, 500);
            fillPoly(masks_snake[j], contour, GC_PR_FGD);

            //DEBUG
            drawGrabcutMask(input, masks_snake[j], tmp, 0.5);
            imshow(window_name, tmp);
            waitKey(2000);
            //TODO remove debug statement
        }

        // Possibly TODO adaptive threshold selection by measuring the area variation of the snake


        ////////// Segmentation with grabcut (mask-initialized) /////////
        for(int j=0; j<boxes.size(); j++) {
            Mat bgm, fgm;
            Rect r = boxes[j];
            grabCut(hsv,
                    masks_snake[j],
                    r,
                    bgm,
                    fgm,
                    8,
                    GC_INIT_WITH_MASK
            );
        }

        for(Mat mask : masks_snake) {
            Mat output{};
            drawGrabcutMask(input, mask, output, 0.5);
            imshow(window_name, output);
            waitKey(2000);
        }
        /*
        for(int index=0; index < masks_gc.size(); index++) {
            Mat final_mask;
            final_mask = (masks_snake[index]==GC_PR_FGD) & (masks_gc[index]==GC_PR_FGD);
            imshow(window_name, 0.5*final_mask+0.5*input_gray);
            waitKey();
        }
         */
    }
}
