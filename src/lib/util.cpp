//
// Created by filippo on 07/07/22.
//

#include "util.h"
#include <fstream>
using namespace cv;
using namespace std;

double IoU_score(Rect detected, Rect ground_truth) {
    Rect I = detected & ground_truth;
    Rect U = detected | ground_truth;
    return I.area()/U.area();
}

double pixel_accuracy(Mat detected, Mat ground_truth) {
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

vector<Rect> extract_bboxes(string txt_path, int padding=0) {
    ifstream boxes_txt = ifstream(txt_path);
    
    if(!boxes_txt.is_open()) {
        cerr << "File not found";
        return vector<Rect>{};
    }

    vector<Rect> boxes;
    while(!boxes_txt.eof()) {
        int x, y, w, h; //params of the rectangle to be read from file
        boxes_txt >> x >> y >> w >> h;
        boxes.push_back(Rect{x+padding, y+padding, w+(2*padding), h+(2*padding)});
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
