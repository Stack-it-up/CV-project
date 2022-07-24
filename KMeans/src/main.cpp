//
// Created by Giorgio on 23/07/22.
//

//Exploiting K-means for segmenting a hand and comparisons with Otsu's method.
//Main issues: 1) Hard to obtain the same cluster for different segmentations using K-means;
//             2) Otsu's method obtains better results sometimes: hard to choose between Otsu's and K-Means;
//Best results obtained exploiting the YCbCr color space (by neglecting the Y component for K-Means, using grayscale of YCbCr for Otsu).
//In case of grayscale image, K-means must consider also the Y component.

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include "../include/kmeans.h"

using namespace cv;


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

        std::vector<Mat> src_patches_kmeans, src_patches_otsu;
        std::vector<Rect> src_boxes;
        Mat res_kmeans = Mat::zeros(src.rows, src.cols, CV_8UC1);
        Mat res_otsu = Mat::zeros(src.rows, src.cols, CV_8UC1);

        for(int l = 0; l < img_boxes.size(); l++) {

            Mat patch_rgb = src(img_boxes[l]).clone();

            Mat patch_ycrcb;
            cvtColor(patch_rgb, patch_ycrcb, COLOR_BGR2YCrCb);

            Mat patch_ycrcb_gray;
            cvtColor(patch_ycrcb, patch_ycrcb_gray, COLOR_BGR2GRAY);

            Mat otsu;
            cv::threshold(patch_ycrcb_gray, otsu, 0, 255, cv::THRESH_OTSU);

            Mat kmeans_ycrcb;
            kmeans_ycrcb = k_means(patch_ycrcb, use_y, 1, 1); //K-means only on Cb and Cr components

            cvtColor(kmeans_ycrcb, kmeans_ycrcb, COLOR_BGR2GRAY);

            src_patches_kmeans.push_back(kmeans_ycrcb);
            src_patches_otsu.push_back(otsu);
            src_boxes.push_back(img_boxes[l]);

            imshow("Original", patch_rgb);
            imshow("Otsu on YCrCb grayscale", otsu);
            imshow("K-means on Cr Cb", kmeans_ycrcb);

            waitKey(0);
        }


        for(int z = 0; z < src_patches_kmeans.size(); z++) {

            src_patches_kmeans.at(z).copyTo(res_kmeans(src_boxes[z]));
            src_patches_otsu.at(z).copyTo(res_otsu(src_boxes[z]));
        }

        cv::imshow("K-means on Image", res_kmeans);
        cv::imshow("Otsu on Image", res_otsu);
        waitKey(0);

    }

    return 0;
}
