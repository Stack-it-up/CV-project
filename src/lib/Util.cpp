//
// Created by filippo on 07/07/22.
//

#include "Util.h"
#include <fstream>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;


double IoU_score(Rect detected, Rect ground_truth) {
    Rect I = detected & ground_truth;
    Rect U = detected | ground_truth;
    return static_cast<double>(I.area())/static_cast<double>(U.area());
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

vector<Rect> extract_bboxes(string const& txt_path, double fractional_padding) {
    CV_Assert(fractional_padding > 0);

    ifstream boxes_txt = ifstream(txt_path);
    if(!boxes_txt.is_open()) {
        cerr << "File not found";
        return vector<Rect>{};
    }

    /*
     * Let fractional_padding = f_p
     * new_area = f_p * area
     * So:
     *
     * new_w = sqrt(f_p) * w
     * new_w = w + 2*pad_x
     *
     * SOLVE FOR pad_x:
     * pad_x = w*(sqrt(f_p)-1)/2
     *
     * similarly for h and pad_y.
     *
     * Note that when pad > 0 we have f_p > 1 (enlarging)
     * when pad < 0 we have f_p < 1 (shrinking)
     */
    vector<Rect> boxes;
    while(!boxes_txt.eof()) {
        int x, y, w, h; //params of the rectangle to be read from file
        boxes_txt >> x >> y >> w >> h;
        int padding_x = cvRound(0.5 * w * (sqrt(fractional_padding)-1));
        int padding_y = cvRound(0.5 * h * (sqrt(fractional_padding)-1));
        boxes.emplace_back(x-padding_x, y-padding_y, w+(2*padding_x), h+(2*padding_y));
    }
    boxes_txt.close();
    return boxes;
}

void show_bboxes(string const& img_path, string const& txt_path) {
    Mat input = imread(img_path);
    vector<Rect> boxes = extract_bboxes(txt_path);

    for(auto const& r : boxes)
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

double avg_IoU_score(vector<Rect> &detected, vector<Rect> &ground_truth, double threshold) {
    struct Score {
        double IoU = 0;
        int det_index = -1;
    };

    double avg_IoU = 0;
    int tp = 0;
    int fp = static_cast<int>(detected.size());
    int fn = 0;

    vector<vector<double>> all_IoU_scores(ground_truth.size());
    vector<Score> IoU_scores(ground_truth.size());

    for (vector<double> &det_vec : all_IoU_scores) {
        det_vec = vector<double>(detected.size());
    }

    // Compute all IoU scores for ground_truth x detected
    for (int i = 0; i < ground_truth.size(); i++) {
        for (int j = 0; j < detected.size(); j++) {
            all_IoU_scores[i][j] = IoU_score(detected[j], ground_truth[i]);
        }
    }

    for (int c_gt = 0; c_gt < ground_truth.size(); c_gt++) {
        double max_IoU = 0;
        int gt_index = -1;
        int det_index = -1;

        // find maximum IoU in matrix all_IoU_scores
        for (int i = 0; i < ground_truth.size(); i++) {
            for (int j = 0; j < detected.size(); j++) {
                if (all_IoU_scores[i][j] > max_IoU) {
                    max_IoU = all_IoU_scores[i][j];
                    gt_index = i;
                    det_index = j;
                }
            }
        }

        if (max_IoU < threshold) {
            break;
        }

        if (IoU_scores[gt_index].det_index == -1) {
            IoU_scores[gt_index].det_index = det_index;
            IoU_scores[gt_index].IoU = all_IoU_scores[gt_index][det_index];

            avg_IoU += IoU_scores[gt_index].IoU;

            fp--;
            tp++;
        }

        all_IoU_scores[gt_index][det_index] = -1;
    }

    fn = static_cast<int>(IoU_scores.size()) - tp;
    avg_IoU /= tp + fp + fn;

    return avg_IoU;
}

bool is_monochromatic(cv::Mat const& input) {
    CV_Assert(input.type() == CV_8UC3);
    Mat hsv;
    cvtColor(input, hsv, COLOR_BGR2HSV_FULL);

    Mat channels[3];
    split(hsv, channels);

    int count0 = countNonZero(channels[0] != channels[0].at<uchar>(0,0));
    int count1 = countNonZero(channels[1] != channels[1].at<uchar>(0,0));

    return (count0 + count1 == 0);
}
