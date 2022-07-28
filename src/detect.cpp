//
// Created by Davide Sarraggiotto on 16/07/2022.
//

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "detector.h"
#include "Util.h"
#include "detect.h"

using namespace std;
using namespace cv;

const string CFG_V3_PATH = "../res/cfg/yolov3-tiny-custom.cfg";
const string CFG_V4_PATH = "../res/cfg/yolov4-tiny-custom.cfg";
const string WEIGHTS_V3_PATH = "../res/cfg/yolov3-tiny-custom.weights";
const string WEIGHTS_V4_PATH = "../res/cfg/yolov4-tiny-custom.weights";
const string IMAGES_PATH = "../res/evaluation_data/rgb/*.jpg";
const string BOUNDING_BOXES_PATH = "../res/evaluation_data/det/*.txt";
const string EXPORT_TXT_PATH = "../exp/det/";
const string EXPORT_IMG_PATH = "../exp/bb_img/";

constexpr float CONF_THRESH = 0.3;
constexpr float NMS_THRESH = 0.4;
constexpr double IOU_THRESH = 0.1;

void hand_detect::detect(const cv::Mat &input, std::vector<cv::Rect> &output_bb, std::vector<float> &output_conf, bool show_image) {
    dnn::Net net_v3 = dnn::readNetFromDarknet(CFG_V3_PATH, WEIGHTS_V3_PATH);
    net_v3.setPreferableTarget(dnn::DNN_TARGET_CPU);
    net_v3.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);

    dnn::Net net_v4 = dnn::readNetFromDarknet(CFG_V4_PATH, WEIGHTS_V4_PATH);
    net_v4.setPreferableTarget(dnn::DNN_TARGET_CPU);
    net_v4.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);

    const vector<dnn::Net> nets {net_v3, net_v4};

    cout << "Detecting hands for the given image\n";

    hand_detect::detect(nets, input, output_bb, output_conf, show_image);
}

void hand_detect::detect(std::vector<cv::dnn::Net> const& nets, const cv::Mat &input, std::vector<cv::Rect> &output_bb, std::vector<float> &output_conf, bool show_image) {
    if (nets.empty()) {
        return;
    }

    hand_detect::detector::detect(nets, input, output_bb, output_conf, CONF_THRESH, NMS_THRESH);
    cout << "- " << output_bb.size() << " hands detected\n";

    if (show_image) {
        hand_detect::detector::show(input.clone(), output_bb);
    }
}

void hand_detect::detection_demo() {
    dnn::Net net_v3 = dnn::readNetFromDarknet(CFG_V3_PATH, WEIGHTS_V3_PATH);
    net_v3.setPreferableTarget(dnn::DNN_TARGET_CPU);
    net_v3.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);

    dnn::Net net_v4 = dnn::readNetFromDarknet(CFG_V4_PATH, WEIGHTS_V4_PATH);
    net_v4.setPreferableTarget(dnn::DNN_TARGET_CPU);
    net_v4.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);

    const vector<dnn::Net> nets {net_v3, net_v4};
    vector<Mat> images;
    vector<std::string> images_names;
    vector<vector<Rect>> original_bounding_boxes;

    loadImages(images, IMAGES_PATH, images_names);
    loadBoundingBoxes(original_bounding_boxes, BOUNDING_BOXES_PATH);

    double IoU = 0;

    for (int i = 0; i < images.size(); i++) {
        Mat& image = images[i];
        vector<Rect> bounding_boxes;
        vector<float> confidences;

        cout << "Detecting hands on image " << images_names[i] << endl;

        hand_detect::detector::detect(nets, image, bounding_boxes, confidences, CONF_THRESH, NMS_THRESH);

        // Print original bounding boxes over image
        Scalar color = Scalar(0,0,255);

        Mat image_bb = image.clone();

        for (Rect& rect : original_bounding_boxes[i]) {
            rectangle(image_bb, rect, color, 2);
        }

        double img_IoU = avg_IoU_score(bounding_boxes, original_bounding_boxes[i], IOU_THRESH);
        cout << "- Average IoU: " << img_IoU << endl;

        hand_detect::detector::show(image_bb, bounding_boxes);
        hand_detect::detector::export_bb(bounding_boxes, EXPORT_TXT_PATH + images_names[i] + ".txt");
        hand_detect::detector::export_image_bb(image_bb, bounding_boxes, EXPORT_IMG_PATH + images_names[i]);

        IoU += img_IoU;
        cout << "\n";
    }

    IoU /= static_cast<double>(images.size());
    cout << "Average IoU over all test images: " << IoU << endl;
}
