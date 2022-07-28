//
// Created by filippo on 24/07/22.
//

#ifndef SNAKES_H
#define SNAKES_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

/**
 * Implements the snakes active contour algorithm as described by Gonzalez and Woods in the book Digital Image Processing.
 * @param contour   Must contain an initial contour with at least 6 points.
 *                  Will contain the final contour once the algorithm terminates.
 * @param ext_force_x Component x of the external force (of type CV_64FC1)
 * @param ext_force_y Component y of the external force (of type CV_64FC1)
 * @param alpha positive coefficient for the elastic (internal) energy
 * @param beta positive coefficient for the bending (internal) energy
 * @param gamma positive coefficient for the external energy
 * @param iters number of iterations
 */
void compute_snake(std::vector<cv::Point>& contour,
                   cv::Mat const& ext_force_x,
                   cv::Mat const& ext_force_y,
                   double alpha, double beta, double gamma,
                   int iters=500);

/**
 * Implements the Magnitude of Gradient calculation
 * to be used as an external force component in active contour algorithms.
 *
 * @param input of type CV_8UC1
 * @param output_x of type CV_64FC1
 * @param output_y idem
 */
void MOG(cv::Mat const& input, cv::Mat& output_x, cv::Mat& output_y);


/**
 * Implements the Vector Field Convolution algorithm (Li and Acton, 2007)
 * to be used as an external force component in active contour algorithms.
 * The magnitude function is m1 = (r+EPSILON)^gamma
 *
 * @param input of type CV_8UC1
 * @param output_x of type CV_64FC1
 * @param output_y of type CV_64FC1
 * @param k kernel size. Note that the complexity of the algorithm does not depend linearly on k, since convolution
 *          is implemented in the frequency domain for k > 11
 * @param gamma
 */
void VFC(cv::Mat const& input, cv::Mat& output_x, cv::Mat& output_y, int k, double gamma);


/**
 * Get the contour from a rect
 *
 *  @param step add a point every step pixels on the line
 */
std::vector<cv::Point> contour_from_rect(cv::Rect bbox, int step=1);
#endif //SNAKES_H
