//
// Created by filippo on 27/07/22.
//

#ifndef HAND_DETECT_SEGMENT_H
#define HAND_DETECT_SEGMENT_H

#include <opencv2/core.hpp>
namespace hand_detect {
    /**
     * Performs segmentation taking as input the bboxes. Shows instance segmentation results and optionally other intermediate steps.
     * @param bboxes_path path to the file of bounding boxes (in the format specified by the project assignment)
     * @param show_steps shows multiple images for each step of the segmentation (original - meanshift - snakes - colored segmentation)
     */
    void segment(cv::Mat const& input,
                 cv::Mat& output,
                 std::string const& bboxes_path,
                 bool show_steps=false);

    /**
     * Perform segmentation on the whole dev dataset and compute pixel accuracy.
     */
    void segmentation_demo();
}
#endif //HAND_DETECT_SEGMENT_H
