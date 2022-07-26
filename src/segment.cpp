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
        constexpr double scale_XXL = 1.0;
        constexpr double scale_XXS = 0.7;
        Mat input = imread(photos_paths[i]);
        vector<Rect> boxes_XXL = extract_bboxes(bbox_paths[i], scale_XXL);
        vector<Rect> boxes_XXS = extract_bboxes(bbox_paths[i], scale_XXS);
        const char* window_name = "win";
        namedWindow(window_name, WINDOW_NORMAL);

        //// Initial segmentation with meanshift in HSV colorspace //////
        Mat hsv{};
        cvtColor(input, hsv, COLOR_BGR2HSV_FULL);
        pyrMeanShiftFiltering(input, hsv, 20, 12, 1);
        imshow(window_name, hsv);
        waitKey(1000);

        //////////////// Mask intialization with snakes //////////////////
        vector<Mat> masks(boxes_XXL.size()); //contains all bounding boxes detected in the image
        Mat input_gray;
        cvtColor(input, input_gray, COLOR_BGR2GRAY);


        //compute the external force using vector field convolution
        Mat vfc_x, vfc_y;
        int ker_size = max(input.rows, input.cols);
        ker_size /= 2;
        if (ker_size % 2 == 0)
            ker_size -= 1;
        VFC(input_gray, vfc_x, vfc_y, ker_size, 2.4);

        for(int j=0; j<boxes_XXL.size(); j++) {
            masks[j] = Mat{input.size(), CV_8UC1, GC_BGD}; //initialize the mask as "definitely background"
            vector<Point> contour = contour_from_rect(boxes_XXL[j]);
            fillConvexPoly(masks[j], contour, GC_PR_BGD); //initialize the rectangle as "probably background"


            //DEBUG
            Mat tmp;
            /*
            drawGrabcutMask(input, masks[j], tmp, 0.5);
            imshow(window_name, tmp);
            waitKey(1000);
            //TODO remove debug statement
             */

            //compute the snake from a smaller rectangle
            contour = contour_from_rect(boxes_XXS[j]);
            compute_snake(contour, vfc_x, vfc_y, 0.8, 0.5, 2, 500);
            fillPoly(masks[j], contour, GC_PR_FGD);

            //DEBUG
            drawGrabcutMask(input, masks[j], tmp, 0.5);
            imshow(window_name, tmp);
            waitKey(1000);
            //TODO remove debug statement
        }


        ////////// Segmentation with grabcut (mask-initialized) /////////
        bool apply_grabcut = true;
        //detect if the image is monochromatic

        if(is_monochromatic(input)) {
            cerr << "[WARN] Input image is monochromatic, segmentation is based on snakes only\n\n";
            apply_grabcut = false;
        }

        if(apply_grabcut) {
            for (int j = 0; j < boxes_XXL.size(); j++) {
                //check if there are too few foreground pixels
                int mode = GC_INIT_WITH_MASK;
                int n_pixels = boxes_XXL[j].area();
                int n_fgd = countNonZero(masks[j] == GC_PR_FGD | masks[j] == GC_FGD);

                if (n_fgd < 0.1 * n_pixels) {
                    mode = GC_INIT_WITH_RECT;
                    cerr << "[WARN] Less than 10% of the pixels in the bbox are identified as foreground! " <<
                    "Proceeding with rectangle intialization...\n\n";
                }

                Mat bgm, fgm;
                Rect r = boxes_XXL[j];
                grabCut(hsv,
                        masks[j],
                        r,
                        bgm,
                        fgm,
                        10,
                        mode
                );
            }
        }

        //keep only the largest-area connected component and join all masks
        Mat final_mask = Mat::zeros(input.size(), CV_8UC3);
        for(Mat& mask : masks) {
            Mat rand_color{input.size(), CV_8UC3, Vec3b{uchar(theRNG()), uchar(theRNG()), uchar(theRNG())}};
            rand_color.copyTo(final_mask, mask==GC_PR_FGD | mask==GC_FGD);
        }
        addWeighted(input, 0.3, final_mask, 0.7, 0, final_mask);
        imshow(window_name, final_mask);
        waitKey(0);
    }
}
