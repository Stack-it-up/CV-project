//
// Created by Giorgio on 20/07/22.
//
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>

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
        std::vector<Rect> img_boxes = boxes[n];

        for(int l = 0; l < img_boxes.size(); l++) {

            Mat patch = src(img_boxes[l]);

            Mat patch_hsv;
            cvtColor(patch, patch_hsv, COLOR_BGR2HSV);

            const int feature_dim = 6;
            cv::Mat points = cv::Mat(patch.rows * patch.cols, feature_dim, CV_32F);

            int m = 0;

            Scalar patch_mean_rgb, patch_stddev_rgb, patch_mean_hsv, patch_stddev_hsv;
            cv::meanStdDev(patch, patch_mean_rgb, patch_stddev_rgb);
            cv::meanStdDev(patch_hsv, patch_mean_hsv, patch_stddev_hsv);

            for (int i = 0; i < patch.rows; i++) {
                for (int j = 0; j < patch.cols; j++) {

                    float b = patch.at<cv::Vec3b>(i, j)[0];
                    float g = patch.at<cv::Vec3b>(i, j)[1];
                    float r = patch.at<cv::Vec3b>(i, j)[2];

                    float h = patch_hsv.at<cv::Vec3b>(i, j)[0];
                    float s = patch_hsv.at<cv::Vec3b>(i, j)[1];
                    float v = patch_hsv.at<cv::Vec3b>(i, j)[2];


                    b = abs(b - patch_mean_rgb[0]) / (patch_stddev_rgb[0] + 0.001);
                    g = abs(g - patch_mean_rgb[1]) / (patch_stddev_rgb[1] + 0.001);
                    r = abs(r - patch_mean_rgb[2]) / (patch_stddev_rgb[2] + 0.001);

                    h = abs(h - patch_mean_hsv[0]) / (patch_stddev_hsv[0] + 0.001);
                    s = abs(s - patch_mean_hsv[1]) / (patch_stddev_hsv[1] + 0.001);
                    v = abs(v - patch_mean_hsv[2]) / (patch_stddev_hsv[2] + 0.001);

                    points.at<cv::Vec<float, feature_dim>>(m) = cv::Vec<float, feature_dim>(
                            b,
                            g,
                            r,
                            h,
                            s,
                            v);

                    m++;
                }
            }

            int k = 2;
            cv::Mat labels;
            cv::TermCriteria tmc = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 25, 0.01);
            int attempts = 5;

            cv::kmeans(points, k, labels, tmc, attempts, cv::KMEANS_PP_CENTERS);

            for (int i = 0; i < labels.rows; i++) {

                int col = i % patch.cols;
                int row = (int) i / patch.cols;

                if (labels.at<int>(i, 0) == 0)
                    patch.at<cv::Vec3b>(row, col) = cv::Vec3b(255, 255, 255);

                else
                    patch.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);
            }

            cv::imshow("KMeans", patch);
            waitKey(0);

        }

        cv::imshow("Image", src);
        waitKey(0);

    }

    return 0;
}
