//
// Created by Davide Sarraggiotto on 16/07/2022.
//


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "detector.h"
#include "Util.h"

using namespace std;
using namespace cv;

void loadImages(vector<Mat>& images, string& folder_path, vector<cv::String>& images_names);

int main() {
    String cfg = "../res/cfg/yolov3-tiny-custom.cfg";
    String weights = "../res/cfg/yolov3-tiny-custom-mono.weights"; //Put the weights file under cfg directory
    String images_path = "../res/evaluation_data/rgb/*.jpg";
    String export_path = "../out/det/";
    String image_export_path = "../out/bb_img/";

    float conf_thresh = 0.3;

    dnn::Net net = dnn::readNetFromDarknet(cfg, weights);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);

    vector<Mat> images;
    vector<cv::String> images_names;
    loadImages(images, images_path, images_names);


    for (int i = 0; i < images.size(); i++) {
        Mat& image = images[i];
        vector<Rect> bounding_boxes;
        vector<float> confidences;

        h_det::detect(net, image, bounding_boxes, confidences, conf_thresh);
        h_det::show(image, bounding_boxes, confidences);
        //h_det::export_bb(bounding_boxes, confidences, export_path + images_names[i] + ".txt", conf_thresh);
        //h_det::export_image_bb(image, bounding_boxes, confidences, image_export_path + images_names[i], conf_thresh);
    }
}

void loadImages(vector<Mat>& images, string& folder_path, vector<cv::String>& images_names) {
    vector<cv::String> img_names;
    glob(folder_path, img_names, false);

    for (String& img_name : img_names) {
        images.push_back(imread(img_name));

        cv::String image_name = img_name.substr(folder_path.size() - 5, img_name.size() - folder_path.size() + 5);
        images_names.push_back(image_name);
    }
}