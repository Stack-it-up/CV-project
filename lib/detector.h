//
// Created by Davide Sarraggiotto on 16/07/2022.
//

#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

namespace hand_detect {

    void detect(std::vector<cv::dnn::Net> &nets, cv::Mat &img, std::vector<cv::Rect> &bounding_boxes, std::vector<float> &confidences,
                float CONF_THRESH = 0.5, float NMS_THRESH = 0.5);

    void show(cv::Mat &img, std::vector<cv::Rect> &bounding_boxes);

    void detect_and_show(std::vector<cv::dnn::Net> &nets, cv::Mat &img, std::vector<cv::Rect> &bounding_boxes, std::vector<float> &confidences,
                         float CONF_THRESH = 0.5, float NMS_THRESH = 0.5);

    void export_bb(std::vector<cv::Rect>& bounding_boxes, const std::string& export_path);

    void export_image_bb(cv::Mat& img, std::vector<cv::Rect>& bounding_boxes, const std::string& export_path);
}

#endif //YOLO_DETECTOR_H