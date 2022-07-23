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

void compute_snake(vector<Point> & initial_contour, Mat const& ext_force_x, Mat const& ext_force_y, double alpha, double beta, double gamma, int iters=500) {
    CV_Assert(initial_contour.size() >= 6); //(otherwise, it won't work for the matrix D4)
    CV_Assert(ext_force_y.type()==CV_64FC1 && ext_force_y.type()==CV_64FC1);
    CV_Assert(alpha>0 && beta>0 && gamma>0);

    int K = initial_contour.size();
    Mat A{};
    create_A(A, K, alpha, beta);

    //create the vectors x and y to be premultiplied later
    //x_t contains all the x's of the points in the contour
    Mat x_t(K, 1, CV_64FC1); //even though only integers should be inside here
    Mat y_t(K, 1, CV_64FC1); //idem
    for(int index=0; index<K; index++) {
        Point p = initial_contour.at(index);
        x_t.at<double>(index) = static_cast<double>(p.x);
        y_t.at<double>(index) = static_cast<double>(p.y);
    }

    //cerr << "K: " << K << "\n";
    Mat Fx = Mat::zeros(K, 1, CV_64FC1);
    Mat Fy = Mat::zeros(K, 1, CV_64FC1);
    for(int t=1; t<iters; t++) {
        //cerr << "t: " << t << "\n";
        //cerr << "x_t size: " << x_t.size() << "\n";
        //cerr << "y_t size: " << y_t.size() << "\n";
        //initialize Fx and Fy
        for(int i=0; i<K; i++) {
            int x_coord = cvRound(x_t.at<double>(i));
            //cerr << "i: " << i << "\n";
            //cerr << "x_coord: " << x_coord << "\n";
            int y_coord = cvRound(y_t.at<double>(i));
            //cerr << "y_coord: " << y_coord << "\n";
            Fx.at<double>(i) = abs(ext_force_x.at<double>(x_coord, y_coord));
            Fy.at<double>(i) = abs(ext_force_y.at<double>(x_coord, y_coord));
        }
        //compute the new vector
        x_t = A * (x_t + gamma*Fx);
        //cerr << A;
        //cerr << x_t << "\n";
        y_t = A * (y_t + gamma*Fy);
        //maybe TODO implement early stopping based on convergence rate?
    }
    //write out the definitive contours
    for(int index=0; index<K; index++) {
        int px = cvRound(x_t.at<double>(index));
        int py = cvRound(y_t.at<double>(index));
        initial_contour[index] = Point{px, py};
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
    namedWindow("win", WINDOW_NORMAL);
    string imgdir = "/home/filippo/Desktop/test/beer.jpeg";
    string imgdir2 = "/home/filippo/Desktop/test/beer2.jpeg";
    Mat beer = imread(imgdir, IMREAD_COLOR);
    Mat beer_gray;
    cvtColor(beer, beer_gray, COLOR_BGR2GRAY);
    Mat beer2 = imread(imgdir2, IMREAD_GRAYSCALE);
    threshold(beer2, beer2, 250, 255, THRESH_BINARY);
    Laplacian(beer2, beer2, CV_8UC1);
    imshow("win", beer2);
    waitKey();

    Mat vfc_x;
    Mat vfc_y;
    int ker_size = min(beer.rows, beer.cols);
    //ker_size /= 2;
    if (ker_size % 2 == 0)
        ker_size -= 1;
    compute_vfc(beer_gray, vfc_x, vfc_y, 201, 2);

    Mat fx;
    Mat fy;
    compute_MOG(beer_gray, fx, fy);
    /*
    vector<Point> fake_snake{};
    for(int i = 0; i<8; i++)
        fake_snake.emplace_back(i,i); //Point() is called by emplace_back!
    fake_snake.emplace_back(0,0); //to make it closed!
    */
    Scalar RED{0,0,255};
    Scalar GREEN{0, 255, 0};

    vector<vector<Point>> contours;
    findContours(beer2, contours, RETR_LIST, CHAIN_APPROX_NONE);
    drawContours(beer, contours, 0, RED);
    imshow("win", beer);
    waitKey();

    compute_snake(contours[0], vfc_x, vfc_y, 6, 0.3, 0.15, 1000);
    drawContours(beer, contours, 0, GREEN);
    imshow("win", beer);
    waitKey();
}
