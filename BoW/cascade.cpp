//
// Created by Davide Sarraggiotto on 08/07/2022.
//

// Let's give it a try

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <random>

using namespace std;
using namespace cv;

void loadImages(vector<Mat>& images, string& folder_path);
void detectAndDisplay(Mat& img, CascadeClassifier& hand_cascade);

int main(int argc, char** argv) {
    CascadeClassifier hand_cascade;

    String hand_cascade_name = samples::findFile("cascade/cascade_lbp48_30stage.xml");

    if (!hand_cascade.load(hand_cascade_name)) {
        cout << "(!) Error loading cascade" << endl;
        return -1;
    }

    if (hand_cascade.isOldFormatCascade()) {
        cout << "(!) Old format cascade detected" << endl;
        return 1;
    }

    cout << "Cascade classifier loaded" << endl;
    cout << "Feature type: " << (hand_cascade.getFeatureType() == 0 ? "HAAR" : "LBP") << endl;
    cout << "Original window size: " << hand_cascade.getOriginalWindowSize().width << "x" << hand_cascade.getOriginalWindowSize().height << endl;

    vector<Mat> test_images;
    string test_images_path = "test/*.jpg";
    loadImages(test_images, test_images_path);

    // std::shuffle(test_images.begin(), test_images.end(), std::mt19937(std::random_device()()));

    for (auto & test_image : test_images) {
        detectAndDisplay(test_image, hand_cascade);
    }

    return 0;
}

void detectAndDisplay(Mat& img, CascadeClassifier& hand_cascade) {
    Mat img_gray;
    Mat img_gray_eq;

    int min_neighbors = 3; // default is 3

    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    equalizeHist(img_gray, img_gray_eq);

    cout << "Detecting hands..." << endl;

    vector<Rect> hands;
    hand_cascade.detectMultiScale(img_gray, hands, 1.1, min_neighbors);

    cout << "Hands detected (not eq): " << hands.size() << endl;

    for (Rect hand : hands) {
        rectangle(img, hand, Scalar(0, 255, 0));
    }

    vector<Rect> hands_eq;
    hand_cascade.detectMultiScale(img_gray_eq, hands_eq, 1.1, min_neighbors);

    cout << "Hands detected (eq): " << hands.size() << endl;

    for (Rect hand : hands_eq) {
        rectangle(img, hand, Scalar(255, 0, 0));
    }

    imshow("Hand detection result (blue for equalized image)", img);
    waitKey(0);
}

void loadImages(vector<Mat>& images, string& folder_path) {
    vector<cv::String> img_names;
    glob(folder_path, img_names, false);

    for (String& img_name : img_names) {
        images.push_back(imread(img_name));
    }
}