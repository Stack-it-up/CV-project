//
// Created by giorgio on 26/07/22.
//
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include "../include/Util.h"
#include "../include/callbacks.h"
#include "../include/snakes.h"
#include "../include/kmeans.h"

using namespace cv;
using namespace std;

int main() {

    std::vector<Mat> all_otsu;
    std::vector<Mat> all_kmeans;
    std::vector<Mat> all_grabcut;

    string photos_dir = "/home/giorgio/University/cv/Project/data/rgb/*.jpg";
    string bboxes_dir = "/home/giorgio/University/cv/Project/data/det/*.txt";

    vector<string> photos_paths;
    vector<string> bbox_paths;
    glob(photos_dir, photos_paths);
    glob(bboxes_dir, bbox_paths);

    sort(photos_paths.begin(), photos_paths.end());
    sort(bbox_paths.begin(), bbox_paths.end());

    /// KMEANS + OTSU

    std::vector<std::vector<Rect>> boxes;
    std::vector<Mat> images;

    for(int i = 0; i < photos_paths.size(); i++) {

        Mat input = imread(photos_paths[i]);
        images.push_back(input);

        std::ifstream boxes_txt;
        boxes_txt.open(bbox_paths[i]);

        std::vector<Rect> img_boxes;

        while (!boxes_txt.eof()) {
            int x, y, w, h;
            boxes_txt >> x >> y >> w >> h;
            img_boxes.push_back(Rect{x, y, w, h});
        }

        boxes.push_back(img_boxes);

        boxes_txt.close();
    }

    for(int n = 0; n < images.size(); n++) {

        Mat src = images[n];

        int count = 0;
        int use_y = 0;

        for(int i = 0; i < src.rows; i++) {
            for(int j = 0; j < src.cols; j++) {

                if(src.at<Vec3b>(i,j)[0] == src.at<Vec3b>(i,j)[1] &&
                   src.at<Vec3b>(i,j)[1] == src.at<Vec3b>(i,j)[2])
                    count++;
            }
        }

        if(count == src.rows * src.cols)
            use_y = 1;

        std::vector<Rect> img_boxes = boxes[n];

        std::vector<Mat> src_patches_kmeans, src_patches_otsu;
        std::vector<Rect> src_boxes;
        Mat res_otsu = Mat::zeros(src.rows, src.cols, CV_8UC1);
        Mat res_kmeans = Mat::zeros(src.rows, src.cols, CV_8UC1);

        for(int l = 0; l < img_boxes.size(); l++) {

            Mat patch_rgb = src(img_boxes[l]).clone();

            Mat patch_ycrcb;
            cvtColor(patch_rgb, patch_ycrcb, COLOR_BGR2YCrCb);

            Mat patch_ycrcb_gray;
            cvtColor(patch_ycrcb, patch_ycrcb_gray, COLOR_BGR2GRAY);

            Mat kmeans;
            kmeans = k_means(patch_ycrcb, use_y, 1, 1, false); //K-means only on Cb and Cr components
            src_patches_kmeans.push_back(kmeans);

            Mat otsu;
            cv::threshold(patch_ycrcb_gray, otsu, 0, 255, cv::THRESH_OTSU);
            src_patches_otsu.push_back(otsu);

            src_boxes.push_back(img_boxes[l]);

        }

        for(int z = 0; z < src_patches_otsu.size(); z++) {
            src_patches_otsu.at(z).copyTo(res_otsu(src_boxes[z]));
            src_patches_kmeans.at(z).copyTo(res_kmeans(src_boxes[z]));
        }

        all_otsu.push_back(res_otsu);
        all_kmeans.push_back(res_kmeans);

        //cv::imshow("OTSU", res_otsu);
        //cv::imshow("KMEANS", res_kmeans);
        //waitKey(0);

    }

    destroyAllWindows();

    /// SNAKES + GRABCUT

    for(int i=0; i<photos_paths.size(); i++) {
        constexpr double scale_XXL = 1.1;
        constexpr double scale_XXS = 0.7;
        Mat input = imread(photos_paths[i]);
        vector<Rect> boxes_XXL = extract_bboxes(bbox_paths[i], scale_XXL);
        vector<Rect> boxes_XXS = extract_bboxes(bbox_paths[i], scale_XXS);
        const char* window_name = "win";
        namedWindow(window_name, WINDOW_NORMAL);

        //// Initial segmentation with meanshift in HSV colorspace //////
        Mat hsv{};
        cvtColor(input, hsv, COLOR_BGR2HSV_FULL);
        pyrMeanShiftFiltering(hsv, hsv, 15, 12, 2);
        //imshow(window_name, hsv);
        //waitKey(0);

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
            //imshow(window_name, tmp);
            //waitKey(1000);
            //TODO remove debug statement

            //compute the snake from a smaller rectangle
            contour = contour_from_rect(boxes_XXS[j]);
            compute_snake(contour, vfc_x, vfc_y, 0.8, 0.5, 2, 500);
            fillPoly(masks[j], contour, GC_PR_FGD);

            //DEBUG
            drawGrabcutMask(input, masks[j], tmp, 0.5);
            //imshow(window_name, tmp);
            //waitKey(1000);
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
                    cerr << "[WARN] Less than 10% of the pixels in the mask are identified as foreground!\n" <<
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

        Mat final_mask = Mat::zeros(input.size(), CV_8UC1);
        for(Mat& mask : masks) { //join all masks
            final_mask = final_mask | (mask==GC_PR_FGD | mask==GC_FGD);
        }

        all_grabcut.push_back(final_mask);

        //morphological refinement
        Mat kernel = getStructuringElement(MORPH_CROSS, Size{5,5});
        morphologyEx(final_mask, final_mask, MORPH_CLOSE, kernel);
        morphologyEx(final_mask, final_mask, MORPH_ERODE, kernel);

        Mat output{input.size(), CV_8UC3, Scalar{0,0,0}};
        Mat green{input.size(), CV_8UC3, Scalar{0,255,0}};
        green.copyTo(output, final_mask);

        addWeighted(input, 0.5, output, 0.5, 0, output);
        //imshow(window_name, output);
        //waitKey();
    }

    //ENSEMBLE TEST

    for(int i = 0; i < all_otsu.size(); i++) {

        Mat km = all_kmeans.at(i);
        Mat ot = all_otsu.at(i);
        Mat gc = all_grabcut.at(i);
        Mat final_patch = Mat(km.rows, km.cols, CV_8UC1);

        for(int n = 0; n < km.rows; n++) {
            for(int m = 0; m < km.cols; m++) {

                unsigned char pixel_km = km.at<unsigned char>(n,m);
                unsigned char pixel_ot = ot.at<unsigned char>(n,m);
                unsigned char pixel_gc = gc.at<unsigned char>(n,m);

                float avg_pixel = (0.2 * pixel_km + 0.2 * pixel_ot + 0.6 * pixel_gc);
                unsigned char final_pixel = 0;

                if(avg_pixel > 127)
                    final_pixel = 255;

                final_patch.at<unsigned char>(n,m) = final_pixel;

            }
        }

        imshow("FINAL PATCH", final_patch);
        waitKey(0);
    }
}