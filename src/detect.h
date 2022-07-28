
// Created by Davide Sarraggiotto on 28/07/2022.
//

#ifndef HAND_DETECT_DETECT_H
#define HAND_DETECT_DETECT_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace hand_detect {
    /**
     * Performs hands detection given an image as input. Optionally, shows the detected bounding boxes over the image.
     * @param input input image
     * @param output_bb vector of detected bounding boxes
     * @param output_conf vector of confidences for each bounding box
     * @param show_image when true shows the input image with detected bounding boxes
     */
    void detect(cv::Mat const& input,
                std::vector<cv::Rect>& output_bb,
                std::vector<float>& output_conf,
                bool show_image=false);

    void detect(std::vector<cv::dnn::Net> const& nets,
                cv::Mat const& input,
                std::vector<cv::Rect>& output_bb,
                std::vector<float>& output_conf,
                bool show_image=false);

    void detection_demo();
}

#endif //HAND_DETECT_DETECT_H
