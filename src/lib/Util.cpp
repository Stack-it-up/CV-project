//
// Created by filippo on 07/07/22.
//

#include "Util.h"
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;


double IoU_score(Rect detected, Rect ground_truth) {
    Rect I = detected & ground_truth;
    Rect U = detected | ground_truth;
    return I.area()/U.area();
}

double pixel_accuracy(Mat& detected, Mat& ground_truth) {
    CV_Assert(detected.type()==CV_8UC1);
    CV_Assert(ground_truth.type()==CV_8UC1);
    CV_Assert(detected.size() == ground_truth.size());

    double count_different = 0;
    for(int row=0; row<detected.rows; row++) {
        for(int col=0; col<detected.cols; col++) {
            if(detected.at<uchar>(row, col) != ground_truth.at<uchar>(row, col))
                count_different += 1;
        }
    }
    return count_different / double(detected.rows * detected.cols);
}

vector<Rect> extract_bboxes(string txt_path, int padding) {
    ifstream boxes_txt = ifstream(txt_path);
    //boxes_txt.open(txt_path);
    if(!boxes_txt.is_open()) {
        cerr << "File not found";
        return vector<Rect>{};
    }

    vector<Rect> boxes;
    while(!boxes_txt.eof()) {
        int x, y, w, h; //params of the rectangle to be read from file
        boxes_txt >> x >> y >> w >> h;
        boxes.push_back(Rect{x-padding, y-padding, w+(2*padding), h+(2*padding)});
    }
    boxes_txt.close();
    return boxes;
}

void show_bboxes(string img_path, string txt_path) {
    Mat input = imread(img_path);
    vector<Rect> boxes = extract_bboxes(txt_path);

    for(Rect r : boxes)
        rectangle(input, r, Scalar{0,0,255});

    imshow("", input);
    waitKey(0);
}

void drawGrabcutMask(Mat& image, Mat& mask, Mat& output, float transparency_level) {
    CV_Assert(transparency_level <= 1 && transparency_level >= 0);
    Scalar FG_COLOR{255, 255, 255};
    Scalar PROB_FG_COLOR{0,255,0};

    Scalar BG_COLOR{0, 0, 255};
    Scalar PROB_BG_COLOR{255,0,0};

    Mat colored_mask{mask.size(), CV_8UC3};

    colored_mask.setTo(BG_COLOR, mask==0);
    colored_mask.setTo(FG_COLOR, mask==1);
    colored_mask.setTo(PROB_BG_COLOR, mask==2);
    colored_mask.setTo(PROB_FG_COLOR, mask==3);

    output.create(image.size(), CV_8UC3);
    addWeighted(image, transparency_level, colored_mask, 1-transparency_level, 0, output);
}

void gradient_mag(cv::Mat& input, cv::Mat& magnitude) {
    CV_Assert(input.type() == CV_8UC1);

    Mat dx = Mat{input.size(), CV_32F};
    Mat dy = dx.clone();
    Mat abs_dx, abs_dy;

    spatialGradient(input, dx, dy);
    Mat mag{dx.size(), CV_32F};
    //use L1 approximation of gradient magnitude
    convertScaleAbs(dx, abs_dx);
    convertScaleAbs(dy, abs_dy);

    addWeighted(abs_dx, 0.5, abs_dy, 0.5, 0, magnitude);
}