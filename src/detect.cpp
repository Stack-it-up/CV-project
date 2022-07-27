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

using namespace std;
using namespace cv;

int main() {
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);

    String cfg_v3 = "../res/cfg/yolov3-tiny-custom.cfg";
    String cfg_v4 = "../res/cfg/yolov4-tiny-custom.cfg";
    String weights_v3 = "../res/cfg/yolov3-tiny-custom.weights";
    String weights_v4 = "../res/cfg/yolov4-tiny-custom.weights";
    String images_path = "../res/evaluation_data/rgb/*.jpg";
    String bounding_boxes_path = "../res/evaluation_data/det/*.txt";
    String export_path = "../exp/det/";
    String image_export_path = "../exp/bb_img/";

    constexpr float conf_thresh = 0.3;
    constexpr float nms_thresh = 0.4;
    constexpr double IoU_thresh = 0.1;

    dnn::Net net_v3 = dnn::readNetFromDarknet(cfg_v3, weights_v3);
    net_v3.setPreferableTarget(dnn::DNN_TARGET_CPU);
    net_v3.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);

    dnn::Net net_v4 = dnn::readNetFromDarknet(cfg_v4, weights_v4);
    net_v4.setPreferableTarget(dnn::DNN_TARGET_CPU);
    net_v4.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);

    vector<dnn::Net> nets {net_v3, net_v4};
    vector<Mat> images;
    vector<std::string> images_names;
    vector<vector<Rect>> original_bounding_boxes;

    loadImages(images, images_path, images_names);
    loadBoundingBoxes(original_bounding_boxes, bounding_boxes_path);

    double IoU = 0;

    for (int i = 0; i < images.size(); i++) {
        Mat& image = images[i];
        vector<Rect> bounding_boxes;
        vector<float> confidences;

        cout << "Detecting hands on image " << images_names[i] << endl;

        h_det::detect(nets, image, bounding_boxes, confidences, conf_thresh, nms_thresh);

        // Print original bounding boxes over image
        Scalar color = Scalar(0,0,255);

        for (int j = 0; j < original_bounding_boxes[i].size(); j++) {
            Rect rect = original_bounding_boxes[i][j];
            rectangle(image, rect, color, 2);
        }

        double img_IoU = avg_IoU_score(bounding_boxes, original_bounding_boxes[i], IoU_thresh);
        cout << "- Average IoU: " << img_IoU << endl;

        //h_det::show(image, bounding_boxes);
        h_det::export_bb(bounding_boxes, export_path + images_names[i] + ".txt");
        h_det::export_image_bb(image, bounding_boxes, image_export_path + images_names[i]);

        IoU += img_IoU;
        cout << "\n";
    }

    IoU /= static_cast<double>(images.size());
    cout << "Average IoU over all test images: " << IoU << endl;
}


