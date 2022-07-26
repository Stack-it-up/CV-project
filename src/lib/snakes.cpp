//
// Created by filippo on 24/07/22.
//

#include "snakes.h"

using namespace cv;
using namespace std;

void create_A(cv::Mat& A, int K, double alpha, double beta) {
    //create the matrix A = [I-D]^-1 to be used in compute_snake()

    //D2 and D4 are matrices of zeros
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

void compute_snake(vector<Point> & contour, Mat const& ext_force_x, Mat const& ext_force_y, double alpha, double beta, double gamma, int iters) {
    CV_Assert(contour.size() >= 6); //(otherwise, it won't work for the matrix D4)
    CV_Assert(ext_force_y.type()==CV_64FC1 && ext_force_y.type()==CV_64FC1);
    CV_Assert(alpha>0 && beta>0 && gamma>0);

    int K = contour.size(); //K is the number of points in the contour
    Mat A{};
    create_A(A, K, alpha, beta);

    //create the vectors x and y to be premultiplied later
    //x_t contains all the x's of the points in the contour
    Mat x_t(K, 1, CV_64FC1); //needed for matrix multiplication, even though only integers should be inside here
    Mat y_t(K, 1, CV_64FC1);
    for(int index=0; index<K; index++) {
        Point p = contour.at(index);
        x_t.at<double>(index) = static_cast<double>(p.x);
        y_t.at<double>(index) = static_cast<double>(p.y);
    }

    Mat Fx = Mat::zeros(K, 1, CV_64FC1);
    Mat Fy = Mat::zeros(K, 1, CV_64FC1);
    for(int t=1; t<iters; t++) {
        //initialize Fx and Fy
        for(int i=0; i<K; i++) {
            int x_coord = cvRound(x_t.at<double>(i));
            if(x_coord < 0)
                x_coord = 0;
            else if(x_coord > ext_force_x.cols)
                x_coord = ext_force_x.cols-1;
            int y_coord = cvRound(y_t.at<double>(i));
            if(y_coord < 0)
                y_coord = 0;
            else if(y_coord > ext_force_x.rows)
                y_coord = ext_force_x.rows-1;


            Fx.at<double>(i) = ext_force_x.at<double>(y_coord, x_coord);
            Fy.at<double>(i) = ext_force_y.at<double>(y_coord, x_coord);
        }
        //compute the new vector
        x_t = A * (x_t + gamma*Fx);
        y_t = A * (y_t + gamma*Fy);
    }
    //write out the final contour
    for(int index=0; index<K; index++) {
        int px = cvRound(x_t.at<double>(index));
        int py = cvRound(y_t.at<double>(index));
        contour[index] = Point{px, py};
    }
}

void MOG(Mat const& input, Mat& output_x, Mat& output_y) {
    Mat blurred;
    blur(input, blurred, Size{7,7});

    Mat dx = Mat{input.size(), CV_64F};
    Mat dy = dx.clone();
    Mat abs_dx, abs_dy;

    Sobel(blurred, dx, CV_64F, 1, 0, 3);
    Sobel(blurred, dy, CV_64F, 0, 1, 3);
    Mat mag{dx.size(), CV_64F};

    //use the squared norm of the gradient vector as energy
    mag = dx.mul(dx) + dy.mul(dy);

    Sobel(mag, dx, CV_64F, 1, 0);
    Sobel(mag, dy, CV_64F, 0, 1);

    //normalize
    Mat F_mag;
    magnitude(dx, dy, F_mag);
    F_mag += 1e-05;
    dx /= F_mag;
    dy /= F_mag;

    //return
    dx.copyTo(output_x);
    dy.copyTo(output_y);
}

void VFC(Mat const& input, Mat& output_x, Mat& output_y, int k, double gamma) {
    CV_Assert(k%2 == 1); //we nee an odd-sized kernel!
    CV_Assert(gamma>0);
    constexpr double EPSILON = 1e-05;

    Mat mag{input.size(), CV_64FC1};
    CV_Assert(input.type() == CV_8UC1);


    Mat dx = Mat{input.size(), CV_64F};
    Mat dy = dx.clone();
    Mat abs_dx, abs_dy;

    Sobel(input, dx, CV_64F, 1, 0);
    Sobel(input, dy, CV_64F, 0, 1);

    //use L1 approximation of gradient magnitude
    magnitude(dx,dy,mag);

    //now that we have the gradient magnitude:
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
    filter2D(mag, output_x, CV_64F, kx);
    filter2D(mag, output_y, CV_64F, ky);

    //normalize the force using L1 norm (cfr. Gonzalez 11-49)
    Mat L2{output_x.size(), CV_64FC1};
    magnitude(output_x, output_y, L2);
    L2 += 1e-05;
    output_x = -output_x / L2;
    output_y = -output_y / L2;
}

vector<Point> contour_from_rect(Rect bbox, int step) {
    //drawing the contour clockwise (NW -> NE -> SE -> SW)
    vector<Point> contour;

    int x,y;
    contour.emplace_back(bbox.x, bbox.y); //add the first point

    x = bbox.x + step;
    while(x <= bbox.x + bbox.width) {
        contour.emplace_back(x, bbox.y);
        x += step;
    }

    x = bbox.x + bbox.width;
    y = bbox.y + step;
    while(y <= bbox.y + bbox.height) {
        contour.emplace_back(x, y);
        y += step;
    }

    x = bbox.x + bbox.width - step;
    y = bbox.y + bbox.height;
    while(x >= bbox.x) {
        contour.emplace_back(x, y);
        x -= step;
    }

    x = bbox.x;
    y = bbox.y + bbox.height - step;
    while(y >= bbox.y) {
        contour.emplace_back(x, y);
        y -= step;
    }

    return contour;
}


