//
// Created by Davide Sarraggiotto on 16/07/2022.
//

#include "detector.h"
#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void hand_detect::detector::detect(std::vector<cv::dnn::Net> const& nets, cv::Mat const& img, std::vector<cv::Rect>& bounding_boxes, std::vector<float>& confidences, float CONF_THRESH, float NMS_THRESH) {
    if (nets.empty()) {
        return;
    }

    if (img.empty()) {
        return;
    }

    int height = img.rows;
    int width = img.cols;

    constexpr double scale = 1.0 / 255.0;
    Size sz = Size(416, 416);
    Mat blob = dnn::blobFromImage(img, scale, sz, Scalar(), true);

    std::vector<Rect> all_bounding_boxes;
    std::vector<float> all_confidences;

    for (dnn::Net net : nets) {
        std::vector<String> layers = net.getLayerNames();
        std::vector<int> indexes = net.getUnconnectedOutLayers();
        std::vector<String> output_layers;

        for (int i = 0; i < std::size(indexes); i++) {

            int index = indexes[i];
            output_layers.push_back(layers[index - 1]);
        }

        net.setInput(blob);
        std::vector<Mat> layer_outputs;

        for (int i = 0; i < size(output_layers); i++) {
            layer_outputs.push_back(net.forward(output_layers[i]));
        }

        for (int i = 0; i < size(layer_outputs); i++) {

            Mat output = layer_outputs.at(i);

            for (int j = 0; j < output.rows; j++) {

                float confidence = output.at<float>(j, 5);

                if (confidence > CONF_THRESH) {

                    float x_center = output.at<float>(j, 0) * width;
                    float y_center = output.at<float>(j, 1) * height;
                    float w = output.at<float>(j, 2) * width;
                    float h = output.at<float>(j, 3) * height;

                    float x = (x_center - w / 2);
                    float y = (y_center - h / 2);

                    Rect bounding_box = Rect(x, y, w, h);

                    all_bounding_boxes.push_back(bounding_box);
                    all_confidences.push_back(confidence);
                }

            }
        }
    }

    std::vector<int> bbox_indexes;
    dnn::NMSBoxes(all_bounding_boxes, all_confidences, CONF_THRESH, NMS_THRESH, bbox_indexes);

    for(int i = 0; i < size(bbox_indexes); i++) {
        int index = bbox_indexes.at(i);
        Rect bounding_box = all_bounding_boxes[index];
        bounding_boxes.push_back(bounding_box);
        confidences.push_back(all_confidences[index]);
    }
}

void hand_detect::detector::show(cv::Mat const& img, std::vector<cv::Rect> const& bounding_boxes) {
    for(int i = 0; i < size(bounding_boxes); i++) {
        Rect bounding_box = bounding_boxes[i];
        Scalar color = Scalar(0,255,0);
        rectangle(img, bounding_box, color, 2);
    }

    imshow("Detection", img);
    waitKey(0);
}

void hand_detect::detector::export_bb(std::vector<cv::Rect> const& bounding_boxes, const std::string& export_path) {
    ofstream output;
    output.open(export_path);

    for(int i = 0; i < size(bounding_boxes); i++) {
        Rect bounding_box = bounding_boxes.at(i);
        output << bounding_box.x << " " << bounding_box.y << " " << bounding_box.width << " " << bounding_box.height << "\n";
    }

    output.close();
}

void hand_detect::detector::export_image_bb(cv::Mat const& img, std::vector<cv::Rect> const& bounding_boxes, const std::string& export_path) {
    for(int i = 0; i < size(bounding_boxes); i++) {
        Rect bounding_box = bounding_boxes.at(i);
        Scalar color = Scalar(0,255,0);
        rectangle(img, bounding_box, color, 2);
    }

    cout << "- Exporting image in " << export_path << "\n";
    imwrite(export_path, img);
}