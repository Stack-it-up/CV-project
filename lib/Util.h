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

/**
 * Computes Intersection over Union score
 * IoU = Area(intersection)/Area(Union)
 */
double IoU_score(cv::Rect detected, cv::Rect ground_truth);

/**
 * Returns pixel accuracy assuming the two images are CV_8UC1 and have the same size.
 * Pixel accuracy = # of equal pixels / # of total pixels
 */
double pixel_accuracy(cv::Mat& detected, cv::Mat& ground_truth);

/**
 * Returns a vector of Rect containing all the bounding boxes in the file at txt_path
 * bboxes are specified in the format used in the assignment
 * @param padding : number of padding pixels around the bounding box
 */
std::vector<cv::Rect> extract_bboxes(std::string txt_path, int padding=0);

/**
 * Shows the image at img_path with the overlay of the bounding boxes specified in the txt file at txt_path.
 * Bounding boxes are drawn in red
 */
void show_bboxes(std::string img_path, std::string txt_path);

/**
 * Draws grabcut mask into output
 */
void drawGrabcutMask(cv::Mat& image, cv::Mat& mask, cv::Mat& output, float transparency_level);

/**
 * Compute gradient magnitude using L1 norm
 * @param input
 * @param magnitude
 */
void gradient_mag(cv::Mat& input, cv::Mat& magnitude);
#endif //UTIL_H
