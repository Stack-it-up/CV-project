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

Mat create_A(int K, double alpha, double beta); //forward declaration

void compute_snake(vector<Point> const& initial_contour, Mat const& Fx, Mat const& Fy, double alpha, double beta, double gamma, int iters=500) {
    int K = initial_contour.size();
    Mat A = create_A(K, alpha, beta);
}

//create the matrix A = [I-D]^-1
Mat create_A(int K, double alpha, double beta) {
    Mat D2 = Mat::zeros(Size{K,K}, CV_64FC1);
    Mat D4 = D2.clone();

    //initialize D2
    for(int row=0; row<K; row++) { //row index and anchor index (point whose value is -2 in the row)
        int prev = (row-1) % K;
        int succ = (row+1) % K;

        D2.at<double>(row, prev) = 1;
        D2.at<double>(row,row) = -2;
        D2.at<double>(row, succ) = 1;
    }

    //similarly, inizialize D4
    for(int row=0; row<K; row++) { //row index and anchor index (point whose value is -2 in the row)
        int prev2 = (row-2) % K;
        int prev = (row-1) % K;
        int succ = (row+1) % K;
        int succ2 = (row+2) % K;

        D4.at<double>(row, prev2) = 1;
        D4.at<double>(row, prev) = -4;
        D4.at<double>(row,row) = 6;
        D4.at<double>(row, succ) = -4;
        D4.at<double>(row, succ2) = 1;
    }

    //weighted sum of the two matrices
    Mat D{Size{K,K}, CV_64FC1};
    addWeighted(D2, alpha, D4, -beta, 0, D);

    //create A as I-D and invert it.
    // I is the identity matrix
    Mat A = Mat::eye(Size{K,K}, CV_64FC1) - D;
    A = A.inv(DECOMP_LU);    //using LU decomposition is highly efficient in this case because D is a 5-banded matrix

    return A;
}

int main() {
    /*
    string DIR = "/home/filippo/Desktop/test/*.jpeg";
    vector<string> imgdirs;
    glob(DIR, imgdirs);

    vector<Mat> imgs(imgdirs.size());
    for(int i=0; i<imgdirs.size(); i++) {
        imgs[i] = imread(imgdirs[i]);
        cvtColor(imgs[i], imgs[i], COLOR_BGR2GRAY);
        //-------------------------------------------------------
        Mat roi{};
        threshold(imgs[i], roi, 150, 255, THRESH_BINARY_INV);
        //-------------------------------------------------------
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours( roi, contours, hierarchy,
                      RETR_LIST, CHAIN_APPROX_SIMPLE );
        cvtColor(roi, roi, COLOR_GRAY2BGR);
        int idx = hierarchy[0][0];
        RNG rand{getTickCount()};
        while(idx >= 0) {
            Scalar color{rand(256), rand(256), rand(256)};
            drawContours(roi, contours, idx, color, 5, 8, hierarchy);
            idx = hierarchy[idx][0];
        }
        imshow("", roi);
        waitKey();

        */

        /*
        Mat vfc_x;
        Mat vfc_y;

        int ker_size = min(imgs[i].rows, imgs[i].cols);
        ker_size /= 2;
        if(ker_size%2 == 0)
            ker_size += 1;
        compute_vfc(imgs[i], vfc_x, vfc_y, 101, 2.4);
         */
        vector<Point> fake_snake{};
        for(int i = 0; i<5; i++)
            fake_snake.emplace_back(i,i); //Point() is called by emplace_back!
        fake_snake.emplace_back(0,0); //to make it closed!

        compute_snake(fake_snake, Mat{}, Mat{}, 0.5, 0.5, 0);
}
