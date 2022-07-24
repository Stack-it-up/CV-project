//
// Created by filippo on 21/07/22.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

//implementation through vector field convolution (Li & Acton, 2007)

using namespace cv;
using namespace std;

void compute_vfc(Mat& input, Mat& output_x, Mat& output_y, int k, double gamma=2.4);

int main() {
    string DIR = "/home/filippo/Desktop/test/*.jpeg";
    vector<string> imgdirs;
    glob(DIR, imgdirs);

    vector<Mat> imgs(imgdirs.size());
    for(int i=0; i<imgdirs.size(); i++) {
        imgs[i] = imread(imgdirs[i]);
        cvtColor(imgs[i], imgs[i], COLOR_BGR2GRAY);
        Mat vfc_x;
        Mat vfc_y;

        int ker_size = min(imgs[i].rows, imgs[i].cols);
        ker_size /= 2;
        if(ker_size%2 == 0)
            ker_size += 1;
        compute_vfc(imgs[i], vfc_x, vfc_y, 101, 2.4);

        namedWindow("vfc_x", WINDOW_NORMAL);
        namedWindow("vfc_y", WINDOW_NORMAL);

        convertScaleAbs(vfc_x, vfc_x);
        convertScaleAbs(vfc_y, vfc_y);

        //equalizeHist(vfc_x, vfc_x);
        //equalizeHist(vfc_y, vfc_y);

        imshow("vfc_x", vfc_x);
        imshow("vfc_y", vfc_y);
        waitKey();
    }


}

//k is the kernel size
void compute_vfc(Mat& input, Mat& output_x, Mat& output_y, int k, double gamma) {
    CV_Assert(k%2 == 1); //we nee an odd-sized kernel!
    CV_Assert(gamma>0);
    constexpr double EPSILON = 1e-05;

    Mat magnitude{};

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
}

