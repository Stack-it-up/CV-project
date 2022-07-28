//
// Created by filippo on 07/07/22.
//

#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace hand_detect {
/**
 * Computes Intersection over Union score
 * IoU = Area(intersection)/Area(Union)
 */
    double IoU_score(cv::Rect detected, cv::Rect ground_truth);

/**
 * Computes average Intersection over Union score
 * @param detected
 * @param ground_truth
 * @param threshold
 */
    double avg_IoU_score(std::vector<cv::Rect> &detected, std::vector<cv::Rect> &ground_truth, double threshold = 0.5);

/**
 * Returns pixel accuracy assuming the two images are CV_8UC1 and have the same size.
 * Pixel accuracy = # of equal pixels / # of total pixels
 */
    double pixel_accuracy(cv::Mat const &detected, cv::Mat const &ground_truth);

/**
 * Returns a vector of Rect containing all the bounding boxes in the file at txt_path.
 * Bounding boxes must be specified in the format specified in the assigned dataset.
 * @param area_scale : if specified, scales the area of rectangle of the specified factor,
 *                      while keeping the center and aspect ratio unmodified.
 *                      Value should be >0 (values >1 are for enlarging, <1 for shrinking).
 */
    std::vector<cv::Rect> extract_bboxes(std::string const &txt_path, double scale_factor = 1.0);

/**
 * Shows the image at img_path with the overlay of the bounding boxes specified in the txt file at txt_path.
 * Bounding boxes are drawn in red
 */
    void show_bboxes(std::string const &img_path, std::string const &txt_path);

/**
 * Draws grabcut mask into output
 */
    void drawGrabcutMask(cv::Mat const &image, cv::Mat const &mask, cv::Mat &output, float transparency_level);

/**
 * Returns true if and only if the image is monochromatic (components H and S are constant)
 * @param input BGR, 8 bit image
 */
    bool is_monochromatic(cv::Mat const &input);

/**
 * Crops the rectangle to make sure it fits inside the image
 */
    void crop_bboxes(cv::Mat const& input, cv::Rect &box);

    void loadImages(std::vector<cv::Mat> &images, std::string const &folder_path, std::vector<std::string> &images_names);

    void loadBoundingBoxes(std::vector<std::vector<cv::Rect>> &bounding_boxes, std::string const &folder_path);
}
#endif //UTIL_H
