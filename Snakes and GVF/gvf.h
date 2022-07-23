//
// Created by filippo on 21/07/22.
//

#ifndef SNAKES_GVF_H
#define SNAKES_GVF_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

using namespace cv; //This is BAAAAD

//k is the kernel size
void compute_vfc(Mat& input, Mat& output_x, Mat& output_y, int k, double gamma) {
    CV_Assert(k%2 == 1); //we nee an odd-sized kernel!
    CV_Assert(gamma>0);
    constexpr double EPSILON = 1e-05;

    Mat magnitude{};
    //TODO use the gradient mag function inside Util (copy-pasted here for convenience)
    CV_Assert(input.type() == CV_8UC1);

    Mat dx = Mat{input.size(), CV_64F};
    Mat dy = dx.clone();
    Mat abs_dx, abs_dy;

    spatialGradient(input, dx, dy);
    Mat mag{dx.size(), CV_64F};

    convertScaleAbs(dx, abs_dx);
    convertScaleAbs(dy, abs_dy);
    //use L1 approximation of gradient magnitude
    addWeighted(abs_dx, 0.5, abs_dy, 0.5, 0, magnitude);
    //-----------------------------------------------

    //now that we have the gradient magnitude:
    //there might be efficiency considerations wrt size of the matrix... see opencv docs for dft()
    Mat nx{Size(k,k), CV_64FC1};
    Mat ny = nx.clone();

    Mat m{Size(k,k), CV_64FC1}; //m1(x,y) = (r+EPSILON)^-gamma

    //fill the nx, ny, m matrices
    for(int x=-k/2; x<=k/2; x++) { //col = x + k/2
        for(int y=-k/2; y<=k/2; y++) {//row = y + k/2
            if(x==0 && y==0) {
                nx.at<double>(k/2,k/2) = 0;
                ny.at<double>(k/2,k/2) = 0;
                continue;
            }
            int col = x + k/2;
            int row = y + k/2;
            double r = sqrt(x*x + y*y);

            nx.at<double>(row, col) = -x/r;
            ny.at<double>(row, col) = -y/r;

            m.at<double>(row, col) = pow((r+EPSILON), -gamma);
        }
    }

    Mat kx{Size(k,k), CV_64FC1};
    Mat ky = kx.clone();

    //elementwise multiplication
    kx = m.mul(nx);
    ky = m.mul(ny);


    //now apply the filtering!! (internally uses the DFT)
    filter2D(magnitude, output_x, CV_64F, kx);
    filter2D(magnitude, output_y, CV_64F, ky);

    //normalize the force using L1 norm (cfr. Gonzalez 11-49)
    output_x = output_x / (abs(output_x)+abs(output_y)+1e-05);
    output_y = output_y / (abs(output_x)+abs(output_y)+1e-05);

}

void compute_MOG(Mat& input, Mat& output_x, Mat& output_y) {
    Mat dx = Mat{input.size(), CV_64F};
    Mat dy = dx.clone();
    Mat abs_dx, abs_dy;

    spatialGradient(input, dx, dy);
    Mat mag{dx.size(), CV_64F};

    //convertScaleAbs(dx, abs_dx);
    //convertScaleAbs(dy, abs_dy);
    //use the squared norm of the gradient vector
    mag = dx.mul(dy) + dy.mul(dy); //elementwise multiplication

    Sobel(mag, dx, CV_64F, 1, 0);
    Sobel(mag, dy, CV_64F, 0, 1);

    //normalize
    mag += 1e-05;
    dx /= mag;
    dy /= mag;

    //return
    dx.copyTo(output_x);
    dy.copyTo(output_y);

    imshow("", dx);
    waitKey();
}
#endif //SNAKES_GVF_H
