//
// Created by Davide Sarraggiotto on 08/07/2022.
//

// Let's give it a try

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include "opencv2/objdetect.hpp"

using namespace std;
using namespace cv;

void loadImages(vector<Mat>& images, string& folder_path);
void detectAndDisplay(Mat& img, CascadeClassifier& hand_cascade);

int main(int argc, char** argv) {
    CascadeClassifier hand_cascade;
    String hand_cascade_name = samples::findFile("cascade/cascade.xml");

    if (!hand_cascade.load(hand_cascade_name)) {
        cout << "(!) Error loading _cascade" << endl;
        return -1;
    }

    vector<Mat> test_images;
    string test_images_path = "test/*.jpg";
    loadImages(test_images, test_images_path);

    std::shuffle(test_images.begin(), test_images.end(), std::mt19937(std::random_device()()));

    for (int i = 0; i < std::min(10, static_cast<int>(test_images.size())); i++) {
        detectAndDisplay(test_images[i], hand_cascade);
    }

    return 0;
}

void detectAndDisplay(Mat& img, CascadeClassifier& hand_cascade) {
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    equalizeHist(img_gray, img_gray);

    cout << "Detecting hands..." << endl;

    vector<Rect> hands;
    hand_cascade.detectMultiScale(img_gray, hands);

    cout << "Hands detected: " << hands.size() << endl;

    for (Rect hand : hands) {
        rectangle(img, hand, Scalar(0, 255, 0));
    }

    imshow("Hand detection result", img);
    waitKey(0);
}

void loadImages(vector<Mat>& images, string& folder_path) {
    vector<cv::String> img_names;
    glob(folder_path, img_names, false);

    for (String& img_name : img_names) {
        images.push_back(imread(img_name));
    }
}