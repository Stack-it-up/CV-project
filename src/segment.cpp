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
    const string PHOTOS_DIR = "../res/evaluation_data/rgb/*.jpg";
    const string BBOXES_DIR = "../exp/det/*.txt";
    const string MASKS_DIR = "../res/evaluation_data/mask/*.png";

    vector<string> photos_paths;
    vector<string> bbox_paths;
    vector<string> masks_paths;
    glob(PHOTOS_DIR, photos_paths);
    glob(BBOXES_DIR, bbox_paths);
    glob(MASKS_DIR, masks_paths);

    //sorting to make sure the files are in the desired order
    sort(photos_paths.begin(), photos_paths.end());
    sort(bbox_paths.begin(), bbox_paths.end());
    sort(masks_paths.begin(), masks_paths.end());

    //scale factors for bounding boxes
    constexpr double scale_XXL = 1.2;
    constexpr double scale_XXS = 0.85;
    double accuracy_accumulator = 0;

    for(int i=0; i<photos_paths.size(); i++) {
        const string window_name = "win";
        namedWindow(window_name, WINDOW_NORMAL);

        const Mat input = imread(photos_paths[i], IMREAD_COLOR);
        const Mat ground_truth_mask = imread(masks_paths[i], IMREAD_GRAYSCALE);
        vector<Rect> boxes_XXL = extract_bboxes(bbox_paths[i], scale_XXL);
        vector<Rect> boxes_XXS = extract_bboxes(bbox_paths[i], scale_XXS);

        //// Initial segmentation with meanshift //////
        Mat rgb{};
        //cvtColor(input, rgb, COLOR_BGR2HSV_FULL);
        pyrMeanShiftFiltering(input, rgb, 20, 12, 1);
        imshow(window_name, rgb);
        //waitKey(1000);

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

            drawGrabcutMask(input, masks[j], tmp, 0.5);
            imshow(window_name, tmp);
            waitKey(1000);
            //TODO remove debug statement

            //compute the snake from a smaller rectangle
            contour = contour_from_rect(boxes_XXS[j]);
            compute_snake(contour, vfc_x, vfc_y, 0.8, 0.5, 2, 600);
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
            cerr << "[WARN] Input image is monochromatic, segmentation is based on snakes only\n";
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
                    cerr << "[WARN] Less than 10% of the pixels in the bbox were identified as foreground: " <<
                    "proceeding with rectangle initialization...\n";
                }

                Mat bgm, fgm;
                Rect r = boxes_XXL[j];
                grabCut(rgb,
                        masks[j],
                        r,
                        bgm,
                        fgm,
                        6,
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

        //compute pixel accuracy
        Mat final_grayscale;
        cvtColor(final_mask, final_grayscale, COLOR_BGR2GRAY);
        threshold(final_grayscale, final_grayscale, 1, 255, THRESH_BINARY);
        cout << "Pixel accuracy for image "<< i <<": "<< pixel_accuracy(final_grayscale, ground_truth_mask) << "\n";
        accuracy_accumulator += pixel_accuracy(final_grayscale, ground_truth_mask);

        //display the final segmentation
        addWeighted(input, 0.3, final_mask, 0.7, 0, final_mask);
        imshow(window_name, final_mask);
        waitKey(500);
    }
    //print the average accuracy on the whole dev set
    accuracy_accumulator /= photos_paths.size();
    cout << "Average pixel accuracy: " << accuracy_accumulator << "\n";
}
