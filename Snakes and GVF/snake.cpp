//
// Created by filippo on 21/07/22.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <iostream>

#include "snakes.h"

using namespace cv;
using namespace std;

int main() {
    namedWindow("win", WINDOW_NORMAL);

    Mat beer = imread("/home/filippo/Desktop/test/hand.jpg", IMREAD_COLOR);
    Mat beer_gray;
    cvtColor(beer, beer_gray, COLOR_BGR2GRAY);
    Mat beer2 = imread("/home/filippo/Desktop/test/hand_cont.png", IMREAD_GRAYSCALE);
    threshold(beer2, beer2, 250, 255, THRESH_BINARY);
    imshow("win", beer2);
    waitKey();

    /*
    Mat circle = imread("/home/filippo/Desktop/test/circle.jpeg", IMREAD_COLOR);
    Mat circle_gray;
    cvtColor(circle, circle_gray, COLOR_BGR2GRAY);
    */
    Mat vfc_x, vfc_y;
    int ker_size = max(beer.rows, beer.cols);
    //ker_size /= 2;
    if (ker_size % 2 == 0)
        ker_size -= 1;
    VFC(beer_gray, vfc_x, vfc_y, 101, 2.4);

    Mat mog_x, mog_y;
    MOG(beer_gray, mog_x, mog_y);

    imshow("mog_x", mog_x);
    imshow("mog_y", mog_y);
    waitKey();

    Scalar RED{0,0,255};
    Scalar GREEN{0, 255, 0};

    vector<vector<Point>> contours;
    findContours(beer2, contours, RETR_LIST, CHAIN_APPROX_NONE);
    drawContours(beer, contours, 0, RED);
    imshow("win", beer);
    waitKey();

    compute_snake(contours[0], mog_x, mog_y, 2, 3, 5, 1000);
    //subsample the contour
    vector<Point> subsampled;
    for(int j=0; j<contours[0].size(); j++) {
        if(j % 2 == 0)
            subsampled.push_back(contours[0][j]);
    }
    contours[0] = subsampled;

    drawContours(beer, contours, 0, GREEN);
    imshow("win", beer);
    waitKey();
}
