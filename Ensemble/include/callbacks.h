#ifndef CALLBACKS_H
#define CALLBACKS_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Util.h"
#include <vector>
/**
 * Data structure for passing data to callback functions
 */
struct Data {
    std::vector<cv::Mat*> mats;
    std::vector<int*> ints;
    const char* winname;

    Data(std::vector<cv::Mat*> m, std::vector<int*> i, const char* win_name)
        :mats(m), ints(i), winname(win_name) {}
};

/**
 *
 *
 */
void bilateral_threshold(int event, void* userdata);

void gaussian_threshold(int event, void* userdata);

void meanshift_trackbar(int event, void* userdata);

#endif //CALLBACKS_H
