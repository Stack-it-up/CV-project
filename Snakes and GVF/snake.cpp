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

    Mat hand = imread("/home/filippo/Desktop/test/hand.jpg", IMREAD_COLOR);
    Mat hand_gray;
    Rect bbox{145, 41, 38, 75};
    bbox += Size{10, 10};
    bbox -= Point(10, 10);
    Rect bbox2{15, 168, 62, 35};
    cvtColor(hand, hand_gray, COLOR_BGR2GRAY);

    imshow("win", hand);
    waitKey();

    //try with VFC
    Mat vfc_x, vfc_y;
    int ker_size = max(hand.rows, hand.cols);
    ker_size /= 2;
    if (ker_size % 2 == 0)
        ker_size -= 1;
    VFC(hand_gray, vfc_x, vfc_y, ker_size, 2.4);

    //try with MOG
    Mat mog_x, mog_y;
    MOG(hand_gray, mog_x, mog_y);

    imshow("mog_x", mog_x);
    imshow("mog_y", mog_y);
    waitKey();

    //plot the initial contour
    Scalar RED{0,0,255};
    Scalar GREEN{0, 255, 0};
    Scalar BLUE {255, 0, 0};

    rectangle(hand, bbox, RED);
    imshow("win", hand);
    waitKey();

    //plot the contour found by snakes
    vector<Point> contour = contour_from_rect(bbox);
    compute_snake(contour, mog_x, mog_y, 2, 3, 5, 1000);

    for(int j=0; j<contour.size()-1; j++) {
        imshow("win", hand);
        waitKey(50);
        line(hand, contour[j], contour[j+1], GREEN);
    }
    line(hand, contour[contour.size()-1], contour[0], GREEN);
    imshow("win", hand);
    waitKey(0);
}
