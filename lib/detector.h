//
// Created by Davide Sarraggiotto on 16/07/2022.
//

#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace hand_detect::detector {

    /**
     * Detects hands on a given image.
     * @param nets vector of networks composing the ensemble
     * @param img input image
     * @param bounding_boxes detected bounding boxes
     * @param confidences detected hands confidences
     * @param CONF_THRESH confidence threshold
     * @param NMS_THRESH non-maximum suppression threshold
     */
    void detect(std::vector<cv::dnn::Net> const& nets, cv::Mat const& img, std::vector<cv::Rect> &bounding_boxes, std::vector<float> &confidences,
                float CONF_THRESH = 0.5, float NMS_THRESH = 0.5);

    /**
     * Show bounding boxes over image
     * @param img
     * @param bounding_boxes
     */
    void show(cv::Mat const& img, std::vector<cv::Rect> const& bounding_boxes);

    /**
     * Export bounding boxes
     * @param bounding_boxes bounding boxes to be exported
     * @param export_path path to file
     */
    void export_bb(std::vector<cv::Rect> const& bounding_boxes, const std::string& export_path);

    /**
     * Export image with bounding boxes
     * @param img original image
     * @param bounding_boxes
     * @param export_path path to image
     */
    void export_image_bb(cv::Mat const& img, std::vector<cv::Rect> const& bounding_boxes, const std::string& export_path);
}

#endif //YOLO_DETECTOR_H