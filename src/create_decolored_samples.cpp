//
// Created by filippo on 20/07/22.
// NB: this algorithm is not made to be run many times, as the images will get progressively darker!
//

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    string directory = "/home/filippo/Desktop/test/*.jpeg";
    vector<string> pics;
    glob(directory, pics);

    for(string s : pics) {
        Mat img = imread(s, IMREAD_COLOR);

        RNG rng = RNG(getTickCount()); //random seed

        if(rng(256) <= 127) { //in expectation 50% of the training set
            cvtColor(img, img, COLOR_BGR2HSV_FULL); //range 0-255 for all components
            uchar hue = uchar(rng);
            uchar sat = uchar(rng);

            for(int row=0; row<img.rows; row++) {
                for(int col=0; col<img.cols; col++) {
                    Vec3b& x = img.at<Vec3b>(row, col);
                    x[0] = hue;
                    x[1] = sat;
                }
            }
            cvtColor(img, img, COLOR_HSV2BGR_FULL);
            waitKey();
            imwrite(s, img);
        }
        else {
            cvtColor(img, img, COLOR_BGR2GRAY); //range 0-255 for all components
            imwrite(s, img);
        }

    }

    return 0;
}

