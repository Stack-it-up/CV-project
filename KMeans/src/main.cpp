//
// Created by Giorgio on 23/07/22.
//

//Exploiting K-means for segmenting a hand and comparisons with Otsu's method.
//Main issues: 1) Hard to obtain the same cluster for different segmentations using K-means (since it is an unsupervised procedure);
//             2) Otsu's method obtains better results sometimes: hard to choose between Otsu's and K-Means;
//Best results obtained exploiting the YCbCr color space (by neglecting the Y component for K-Means, using grayscale of YCbCr for Otsu).
//
//In case of grayscale image, K-means must consider also the Y component.
//By analyzing the Y component of YCbCr color space, it is possible to select between Otsu and K-Means segmentation. This seems to lead to
//better results in general.
//

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include "../include/kmeans.h"

using namespace cv;

bool use_kmeans(Mat y) {

    float grad_x_1 = 0;
    float grad_x_2 = 0;
    float grad_x_3 = 0;
    float grad_y_1 = 0;
    float grad_y_2 = 0;
    float grad_y_3 = 0;
    float grad_global = 0;

    for(int i = 0; i < y.rows; i++) {
        for(int j = 0; j < y.cols; j++) {

            float pixel = y.at<unsigned char>(i,j);

            grad_global = grad_global + pixel;

            if(i < y.rows/3)
                grad_y_1 = grad_y_1 + pixel;
            else if(i >= y.rows/3 && i < 2 * y.rows/3)
                grad_y_2 = grad_y_2 + pixel;
            else
                grad_y_3 = grad_y_3 + pixel;

            if(j < y.cols/3)
                grad_x_1 = grad_x_1 + pixel;
            else if(j >= y.cols/3 && j < 2 * y.cols/3)
                grad_x_2 = grad_x_2 + pixel;
            else
                grad_x_3 = grad_x_3 + pixel;
        }
    }

    grad_x_1 = 3 * grad_x_1 / (y.rows * y.cols);
    grad_x_2 = 3 * grad_x_2 / (y.rows * y.cols);
    grad_x_3 = 3 * grad_x_3 / (y.rows * y.cols);

    grad_y_1 = 3 * grad_y_1 / (y.rows * y.cols);
    grad_y_2 = 3 * grad_y_2 / (y.rows * y.cols);
    grad_y_3 = 3 * grad_y_3 / (y.rows * y.cols);

    float grad_x = abs(grad_x_1 - grad_x_2) + abs(grad_x_2 - grad_x_3) + abs(grad_x_1 - grad_x_3);
    float grad_y = abs(grad_y_1 - grad_y_2) + abs(grad_y_2 - grad_y_3) + abs(grad_y_1 - grad_y_3);

    if(abs(grad_x - grad_y) > 50)
        return true;

    return false;

}


int main(int argc, char ** argv) {

    std::string photos_dir = "/home/giorgio/University/cv/Project/data/rgb/*.jpg";
    std::string bboxes_dir = "/home/giorgio/University/cv/Project/data/det/*.txt";

    std::vector<std::string> photos_paths;
    std::vector<std::string> bbox_paths;
    glob(photos_dir, photos_paths);
    glob(bboxes_dir, bbox_paths);

    sort(photos_paths.begin(), photos_paths.end());
    sort(bbox_paths.begin(), bbox_paths.end());

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

        std::vector<Mat> src_patches;
        std::vector<Rect> src_boxes;
        Mat res = Mat::zeros(src.rows, src.cols, CV_8UC1);

        for(int l = 0; l < img_boxes.size(); l++) {

            Mat patch_rgb = src(img_boxes[l]).clone();

            Mat patch_ycrcb;
            cvtColor(patch_rgb, patch_ycrcb, COLOR_BGR2YCrCb);

            std::vector<Mat> split_ycbcr;

            split(patch_ycrcb, split_ycbcr);

            bool need_kmeans = use_kmeans(split_ycbcr[0]);

            Mat patch_ycrcb_gray;
            cvtColor(patch_ycrcb, patch_ycrcb_gray, COLOR_BGR2GRAY);

            if(need_kmeans) {

                Mat kmeans;
                kmeans = k_means(patch_ycrcb, use_y, 1, 1, false); //K-means only on Cb and Cr components
                if(countNonZero(kmeans) > 0.8 * patch_ycrcb.rows * patch_ycrcb.cols)
                    kmeans = k_means(patch_rgb, 1, 1, 1, true);
                src_patches.push_back(kmeans);
            }
            else {
                Mat otsu;
                cv::threshold(patch_ycrcb_gray, otsu, 0, 255, cv::THRESH_OTSU);
                src_patches.push_back(otsu);
            }

            src_boxes.push_back(img_boxes[l]);

        }

        for(int z = 0; z < src_patches.size(); z++)
            src_patches.at(z).copyTo(res(src_boxes[z]));


        cv::imshow("Result", res);
        waitKey(0);

    }

    return 0;
}
