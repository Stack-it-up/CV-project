#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include "Util.h"
#include "snakes.h"
#include "segment.h"

using namespace cv;
using namespace hand_detect;
using std::string;
using std::vector;
using std::cout;
using std::cerr;

void hand_detect::segment(Mat const& input, Mat& output, string const& bboxes_path, bool show_steps) {
    const string winname = "Segmentation...";
    namedWindow(winname, WINDOW_NORMAL);

    //scale factors for bounding boxes
    constexpr double scale_XXL = 4;
    constexpr double scale_XXS = 0.7;

    ////// Extract and save bounding boxes ////////
    vector<Rect> boxes_M = extract_bboxes(bboxes_path);
    vector<Rect> boxes_XXL = extract_bboxes(bboxes_path, scale_XXL);
    vector<Rect> boxes_XXS = extract_bboxes(bboxes_path, scale_XXS);

    //we need to crop the bounding boxes
    for(Rect& box : boxes_M)
        crop_bboxes(input, box);
    for(Rect& box : boxes_XXL)
        crop_bboxes(input, box);
    for(Rect& box : boxes_XXS)
        crop_bboxes(input, box);
    //// Initial segmentation with meanshift //////
    Mat rgb{};
    pyrMeanShiftFiltering(input, rgb, 20, 12, 1);
    if(show_steps) {
        imshow(winname, input);
        waitKey(1000);
        imshow(winname, rgb);
        waitKey(1000);
    }

    //////////////// Mask intialization with snakes //////////////////
    vector<Mat> masks(boxes_M.size()); //will contain one mask for each bounding box
    Mat input_gray;
    cvtColor(input, input_gray, COLOR_BGR2GRAY);

    //compute the external force using vector field convolution
    Mat vfc_x, vfc_y;
    int ker_size = max(input.rows, input.cols);
    ker_size /= 2;
    if (ker_size % 2 == 0)
        ker_size -= 1;
    VFC(input_gray, vfc_x, vfc_y, ker_size, 2.4);

    //compute the snake for each bounding box and initialize the mask with foreground and background markers
    for(int j=0; j<boxes_M.size(); j++) {
        masks[j] = Mat{input.size(), CV_8UC1, GC_BGD}; //initialize the mask as "definitely background"
        vector<Point> contour = contour_from_rect(boxes_M[j]); //obtain contour of the big rectangle to fill it
        fillConvexPoly(masks[j], contour, GC_PR_BGD); //fill it as "probably background"

        if(show_steps) {
            Mat tmp;
            drawGrabcutMask(input, masks[j], tmp, 0.5);
            imshow(winname, tmp);
            waitKey(1000);
        }

        //compute the snake from the downscaled smaller rectangle
        contour = contour_from_rect(boxes_XXS[j]);
        compute_snake(contour, vfc_x, vfc_y, 0.8, 0.5, 2, 600);
        fillPoly(masks[j], contour, GC_PR_FGD); //initialize the inside of the snake as "probably foreground"

        if(show_steps) {
            Mat tmp;
            drawGrabcutMask(input, masks[j], tmp, 0.5);
            imshow(winname, tmp);
            waitKey(1000);
        }
    }

    ///////// Segmentation with grabcut (mask-initialized) /////////
    //if the image is monochromatic, segmentation will only be based on snakes
    bool apply_grabcut = true;
    if(is_monochromatic(input)) {
        cerr << "[WARN] Input image is monochromatic, segmentation is based on snakes only\n";
        apply_grabcut = false;
    }

    if(apply_grabcut) {
        for (int j = 0; j < boxes_M.size(); j++) {
            //if there are too few foreground pixels, initialize grabcut with the rectangle instead
            Rect r = boxes_M[j];
            int grabcut_mode = GC_INIT_WITH_MASK;
            int n_pixels = boxes_M[j].area();
            int n_fgd = countNonZero(masks[j] == GC_PR_FGD | masks[j] == GC_FGD);
            if (n_fgd < 0.1 * n_pixels) {
                grabcut_mode = GC_INIT_WITH_RECT;
                //adjust the rectangle r
                r.x -= boxes_XXL[j].x;
                r.y -= boxes_XXL[j].y;
                cerr << "[WARN] Less than 10% of the pixels in the bbox were identified as foreground: " <<
                     "proceeding with rectangle initialization...\n";

            }
            //finally, apply grabcut to each mask
            Mat bgm, fgm;

            grabCut(rgb(boxes_XXL[j]),
                    masks[j](boxes_XXL[j]),
                    r,
                    bgm,
                    fgm,
                    7,
                    grabcut_mode
            );
        }
    }

    //join all detected foregrounds using different colors
    Mat final_mask = Mat::zeros(input.size(), CV_8UC3);
    for(Mat& mask : masks) {
        Mat rand_color{input.size(), CV_8UC3, Vec3b{200, uchar(theRNG()), uchar(theRNG())}};
        rand_color.copyTo(final_mask, mask==GC_PR_FGD | mask==GC_FGD);
    }

    //compute the monochromatic mask
    cvtColor(final_mask, output, COLOR_BGR2GRAY);
    threshold(output, output, 1, 255, THRESH_BINARY);

    //show the final segmentation output
    addWeighted(input, 0.3, final_mask, 0.7, 0, final_mask);
    imshow(winname, final_mask);
    waitKey(1000);
}

void hand_detect::segmentation_demo() {
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

    double accuracy_accumulator = 0;

    for(int i=15; i<photos_paths.size(); i++) {
        const Mat input = imread(photos_paths[i], IMREAD_COLOR);
        const Mat ground_truth_mask = imread(masks_paths[i], IMREAD_GRAYSCALE);
        Mat output;
        segment(input, output, bbox_paths[i], true);
        cout << "Pixel accuracy for image "<< i+1 <<": "<< pixel_accuracy(output, ground_truth_mask) << "\n";
        accuracy_accumulator += pixel_accuracy(output, ground_truth_mask);
    }
    accuracy_accumulator /= photos_paths.size();
    cout << "Average pixel accuracy: " << accuracy_accumulator << "\n";
}
