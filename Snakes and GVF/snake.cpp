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
using std::vector;

//TODO rimuovere variabili globali
const int maxvalue = 100;
int alpha_slider=1, beta_slider=1, gamma_slider=1;
double alpha;
double beta;
double g;
Mat hand, vfc_x, vfc_y;
Mat dst;
vector<Point> contour;
Rect bbox;

static void on_trackbar( int, void* )
{
    contour = contour_from_rect(bbox);
    alpha = (double) alpha_slider/maxvalue ;
    beta = (double) beta_slider/maxvalue;
    g = (double) gamma_slider/maxvalue;

    dst = hand.clone();
    compute_snake(contour, vfc_x, vfc_y, alpha, beta, g, 800);
    for(int j=0; j<contour.size()-1; j++) {
        line(dst, contour[j], contour[j+1], Scalar{0,255,0});
    }
    line(dst, contour[contour.size()-1], contour[0], {0,255,0});
    imshow( "win", dst );
}

int main() {
    namedWindow("win", WINDOW_NORMAL);

    hand = imread("../hand.jpg", IMREAD_COLOR);

    //bbox=Rect{145, 41, 38, 75};
    bbox=Rect{15, 168, 62, 35};
    bbox += Size{10, 10};
    bbox -= Point(10, 10);


    imshow("win", hand);
    waitKey();

    Mat hand_gray;
    cvtColor(hand, hand_gray, COLOR_BGR2GRAY);
    //try with VFC
    //Mat vfc_x, vfc_y;
    int ker_size = max(hand.rows, hand.cols);
    ker_size /= 2;
    if (ker_size % 2 == 0)
        ker_size -= 1;
    VFC(hand_gray, vfc_x, vfc_y, ker_size, 2.4);

    /*
    //try with MOG
    Mat mog_x, mog_y;
    MOG(hand_gray, mog_x, mog_y);
     */

    imshow("vfc_x", vfc_x);
    imshow("vfc_y", vfc_y);
    waitKey();

    //plot the initial contour
    Scalar RED{0,0,255};
    Scalar GREEN{0, 255, 0};
    Scalar BLUE {255, 0, 0};

    rectangle(hand, bbox, RED);
    imshow("win", hand);
    waitKey();

    //plot the contour found by snakes
    contour = contour_from_rect(bbox);
    /*
    compute_snake(contour, vfc_x, vfc_y, 0.8, 0.01, 5, 200);

    for(int j=0; j<contour.size()-1; j++) {
        line(hand, contour[j], contour[j+1], GREEN);
    }
    line(hand, contour[contour.size()-1], contour[0], GREEN);
    imshow("win", hand);
    waitKey(0);
     */
    createTrackbar("alpha", "win", &alpha_slider, 10*maxvalue, on_trackbar);
    createTrackbar("bets", "win", &beta_slider, 10*maxvalue, on_trackbar);
    createTrackbar("gamma", "win", &gamma_slider, 10*maxvalue, on_trackbar);
    waitKey();
}
