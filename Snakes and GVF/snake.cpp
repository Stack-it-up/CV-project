//
// Created by filippo on 21/07/22.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <iostream>

#include "gvf.h"

using namespace cv;
using namespace std;

void create_A(cv::Mat& A, int K, double alpha, double beta); //forward declaration

void compute_snake(vector<Point> const& initial_contour, Mat const& ext_force_x, Mat const& ext_force_y, double alpha, double beta, double gamma, int iters=500) {
    CV_Assert(initial_contour.size() >= 6); //(otherwise, it won't work for the matrix D4)
    CV_Assert(ext_force_y.type()==CV_64FC1 && ext_force_y.type()==CV_64FC1);
    CV_Assert(alpha>0 && beta>0 && gamma>0);

    int K = initial_contour.size() - 1;
    Mat A{};
    create_A(A, K, alpha, beta);

    //create the vectors x and y to be premultiplied later
    //x_t contains all the x's of the points in the contour
    Mat x_t{K, 1, CV_16UC1};
    Mat y_t{K, 1, CV_16UC1};
    for(int index=0; index<K; index++) {
        Point p = initial_contour[index];
        x_t.at<ushort>(index) = p.x;
        y_t.at<ushort>(index) = p.y;
    }

    for(int t=1; t<iters; t++) {
        //create and initialize Fx and Fy
        Mat Fx{K, 1, CV_64FC1};
        Mat Fy{K, 1, CV_64FC1};
        for(int i=0; i<K; i++) {
            int x_coord = x_t.at<ushort>(i);
            int y_coord = y_t.at<ushort>(i);
            Fx.at<double>(i) = ext_force_x.at<double>(x_coord, y_coord);
            Fy.at<double>(i) = ext_force_y.at<double>(x_coord, y_coord);
        }
        //compute the new vector
        x_t = A * (x_t + gamma*Fx);
        y_t = A * (y_t + gamma*Fy);
        //maybe TODO implement early stopping based on convergence rate?
    }
}

//create the matrix A = [I-D]^-1
void create_A(cv::Mat& A, int K, double alpha, double beta) {
    Mat D2{Size{K,K}, CV_64FC1, 0.0};
    Mat D4{Size{K,K}, CV_64FC1, 0.0};

    //initialize D2
    for(int row=0; row<K; row++) { //row index and anchor index (point whose value is -2 in the row)
        int prev = (row-1+K) % K;
        int succ = (row+1) % K;

        D2.at<double>(row, prev) = 1;
        D2.at<double>(row,row) = -2;
        D2.at<double>(row, succ) = 1;
    }

    //similarly, inizialize D4
    for(int row=0; row<K; row++) { //row index and anchor index (point whose value is 6 in the row)
        int prev2 = (row-2+K) % K;
        int prev = (row-1+K) % K;
        int succ = (row+1) % K;
        int succ2 = (row+2) % K;

        D4.at<double>(row, prev2) = 1;
        D4.at<double>(row, prev) = -4;
        D4.at<double>(row,row) = 6;
        D4.at<double>(row, succ) = -4;
        D4.at<double>(row, succ2) = 1;
    }


    //weighted sum of the two matrices
    Mat D = alpha*D2 - beta*D4;

    //create A as I-D and invert it.
    // I is the identity matrix
    Mat tmp = Mat::eye(Size{K,K}, CV_64FC1) - D;
    //cout << A <<'\n';
    tmp = tmp.inv(DECOMP_LU);    //using LU decomposition is highly efficient in this case because D is a 5-banded matrix

    tmp.copyTo(A);
}

int main() {

    string DIR = "/home/filippo/Desktop/test/*.jpeg";
    vector<string> imgdirs;
    glob(DIR, imgdirs);

    vector<Mat> imgs(imgdirs.size());
    for (int i = 1; i < imgdirs.size(); i++) {
        imgs[i] = imread(imgdirs[i]);
        cvtColor(imgs[i], imgs[i], COLOR_BGR2GRAY);
        //-------------------------------------------------------
        Mat roi{};
        threshold(imgs[i], roi, 150, 255, THRESH_BINARY_INV);
        //-------------------------------------------------------
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(roi, contours, hierarchy,
                     RETR_LIST, CHAIN_APPROX_SIMPLE);
        cvtColor(roi, roi, COLOR_GRAY2BGR);
        int idx = hierarchy[0][0];
        RNG rand{getTickCount()};
        /*
        while(idx >= 0) {
            Scalar color{rand(256), rand(256), rand(256)};
            drawContours(roi, contours, idx, color, 5, 8, hierarchy);
            idx = hierarchy[idx][0];
        }
        imshow("", roi);
        waitKey();
         */


        Mat vfc_x;
        Mat vfc_y;
        int ker_size = min(imgs[i].rows, imgs[i].cols);
        ker_size /= 2;
        if (ker_size % 2 == 0)
            ker_size += 1;
        compute_vfc(imgs[i], vfc_x, vfc_y, 101, 2.4);

        /*
        vector<Point> fake_snake{};
        for(int i = 0; i<8; i++)
            fake_snake.emplace_back(i,i); //Point() is called by emplace_back!
        fake_snake.emplace_back(0,0); //to make it closed!
        */

        compute_snake(contours[2], vfc_x, vfc_y, 0.5, 0.5, 0.5);
        Scalar color{rand(256), rand(256), rand(256)};
        drawContours(roi, contours[2], -1, color);
    }
}
