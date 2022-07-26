//
// Created by giorgio on 23/07/22.
//
#include "../include/kmeans.h"
#include <opencv2/imgproc.hpp>

using namespace cv;

Mat k_means(Mat img, float a, float b, float c, bool normalize) {

    Mat patch = img.clone();

    const int feature_dim = 3;
    cv::Mat points = cv::Mat(patch.rows * patch.cols, feature_dim, CV_32F);

    int m = 0;

    Scalar mean, std_dev;

    meanStdDev(img, mean, std_dev);

    for (int i = 0; i < patch.rows; i++) {
        for (int j = 0; j < patch.cols; j++) {

            float x = patch.at<cv::Vec3b>(i, j)[0];
            float y = patch.at<cv::Vec3b>(i, j)[1];
            float z = patch.at<cv::Vec3b>(i, j)[2];

            if(normalize) {

                float eps = 0.001;

                x = abs(x - mean[0]) / (std_dev[0] + eps);
                y = abs(y - mean[1]) / (std_dev[1] + eps);
                z = abs(z - mean[2]) / (std_dev[2] + eps);

            }


            points.at<cv::Vec<float, feature_dim>>(m) = cv::Vec<float, feature_dim>(
                    x * a,
                    y * b,
                    z * c);

            m++;

        }
    }

    cv::Mat labels;
    cv::TermCriteria tmc = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 25, 0.01);
    int attempts = 5;
    int clusters = 2;

    cv::kmeans(points, clusters, labels, tmc, attempts, cv::KMEANS_PP_CENTERS);

    int hand = 0;
    int bkg = 0;

    for (int i = 0; i < labels.rows; i++) {

        int col = i % patch.cols;
        int row = (int) i / patch.cols;

        int idx = labels.at<int>(i, 0);

        if(row >= 0.4 * img.rows && row <= 0.6 * img.rows &&
           col >= 0.4 * img.cols && col <= 0.6 * img.cols) {

            if(idx == 0)
                hand++;
            else
                bkg++;
        }

        if(idx == 0) {
            patch.at<cv::Vec3b>(row, col) = Vec3b(255,255,255);
        }
        else {
            patch.at<cv::Vec3b>(row, col) = Vec3b(0,0,0);
        }

    }

    cvtColor(patch, patch, COLOR_BGR2GRAY);

    if(bkg > hand)
        patch = 255 - patch;

    return patch;
}
