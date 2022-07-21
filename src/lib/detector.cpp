//
// Created by Davide Sarraggiotto on 16/07/2022.
//

#include "detector.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

void h_det::detect(cv::dnn::Net& net, cv::Mat& img, std::vector<cv::Rect>& bounding_boxes, std::vector<float>& confidences, float CONF_THRESH) {
    if (net.empty()) {
        return;
    }

    if (img.empty()) {
        return;
    }

    std::vector<String> layers = net.getLayerNames();
    std::vector<int> indexes = net.getUnconnectedOutLayers();
    std::vector<String> output_layers;

    for(int i = 0; i < std::size(indexes); i++) {

        int index = indexes[i];
        output_layers.push_back(layers[index - 1]);
    }

    int height = img.rows;
    int width = img.cols;

    double scale = 0.00392;
    Size sz = Size(416,416);
    Mat blob = dnn::blobFromImage(img, scale, sz, Scalar(), true);
    net.setInput(blob);

    std::vector<Mat> layer_outputs;

    for(int i = 0; i < size(output_layers); i++) {

        layer_outputs.push_back(net.forward(output_layers[i]));
    }

    for(int i = 0; i < size(layer_outputs); i++) {

        Mat output = layer_outputs.at(i);

        for(int j = 0; j < output.rows; j++) {

            float confidence = output.at<float>(j,5);

            if(confidence > CONF_THRESH) {

                float x_center = output.at<float>(j,0) * width;
                float y_center = output.at<float>(j,1) * height;
                float w = output.at<float>(j,2) * width;
                float h = output.at<float>(j,3) * height;

                float x = (x_center - w / 2);
                float y = (y_center - h / 2);

                Rect bounding_box = Rect(x, y,w,h);

                bounding_boxes.push_back(bounding_box);
                confidences.push_back(confidence);
            }

        }
    }
}

void h_det::show(cv::Mat &img, std::vector<cv::Rect> &bounding_boxes, std::vector<float> &confidences, float CONF_THRESH, float NMS_THRESH) {
    std::vector<int> bbox_indexes;

    dnn::NMSBoxes(bounding_boxes, confidences, CONF_THRESH, NMS_THRESH, bbox_indexes);

    for(int i = 0; i < size(bbox_indexes); i++) {
        int index = bbox_indexes.at(i);
        Rect bounding_box = bounding_boxes.at(index);
        Scalar color = Scalar(0,255,0);
        rectangle(img, bounding_box, color, 2);
    }

    imshow("Detection", img);
    waitKey(0);
}

void h_det::detect_and_show(cv::dnn::Net &net, cv::Mat &img, std::vector<cv::Rect> &bounding_boxes, std::vector<float> &confidences, float CONF_THRESH, float NMS_THRESH) {
    h_det::detect(net, img, bounding_boxes, confidences, CONF_THRESH);
    h_det::show(img, bounding_boxes, confidences, CONF_THRESH, NMS_THRESH);
}

void h_det::export_bb(std::vector<cv::Rect>& bounding_boxes, std::vector<float>& confidences, const std::string& export_path, float CONF_THRESH, float NMS_THRESH) {
    std::vector<int> bbox_indexes;
    ofstream output;
    output.open (export_path);

    dnn::NMSBoxes(bounding_boxes, confidences, CONF_THRESH, NMS_THRESH, bbox_indexes);

    for(int i = 0; i < size(bbox_indexes); i++) {
        int index = bbox_indexes.at(i);
        Rect bounding_box = bounding_boxes.at(index);
        output << bounding_box.x << " " << bounding_box.y << " " << bounding_box.width << " " << bounding_box.height << "\n";
    }

    output.close();
}

void h_det::export_image_bb(cv::Mat& img, std::vector<cv::Rect>& bounding_boxes, std::vector<float>& confidences, const std::string& export_path, float CONF_THRESH, float NMS_THRESH) {
    std::vector<int> bbox_indexes;

    dnn::NMSBoxes(bounding_boxes, confidences, CONF_THRESH, NMS_THRESH, bbox_indexes);

    for(int i = 0; i < size(bbox_indexes); i++) {
        int index = bbox_indexes.at(i);
        Rect bounding_box = bounding_boxes.at(index);
        Scalar color = Scalar(0,255,0);
        rectangle(img, bounding_box, color, 2);
    }

    cout << "-Saving image in " << export_path << endl;
    imwrite(export_path, img);
}