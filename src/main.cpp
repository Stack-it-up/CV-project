//
// Created by Davide Sarraggiotto on 28/07/2022.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "detect.h"
#include "segment.h"
#include "detector.h"

using namespace std;
using namespace hand_detect;

void print_help() {
    cout << "usage: hand_detect [-d show_image] [-s show_steps] [-image path_to_image] [-bb path_to_bb]\n";
    cout << "-when no argument is provided, hand_detect runs a demo of both detection and segmentation over the test set\n";
    cout << "-when argument [-d show_image] is provided, hand_detect performs detection over image in path_to_image and exports the detected bounding boxes in path_to_bb. If path_to_image or path_to_bb is not specified, it runs on the test set instead\n";
    cout << "-when argument [-s show_steps] is provided, hand_detect performs segmentation over image in path_to_image and bounding boxes in path_to_bb. If path_to_image or path_to_bb is not specified, it runs on the test set instead\n\n";
}

string_view get_option(const vector<std::string_view>& args, const string_view& option_name) {
    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == option_name)
            if (it + 1 != end)
                return *(it + 1);
    }

    return "";
}

bool has_option(const vector<string_view>& args, const string_view& option_name) {
    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == option_name)
            return true;
    }

    return false;
}

bool is_number(const std::string& s) {
    return !s.empty() && std::find_if(s.begin(),s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

bool parse(int argc, char* argv[], bool &detection, bool &segmentation, bool &show_img, bool &show_steps, string &path_to_image, string &path_to_bb) {
    const vector<string_view> args(argv + 1, argv + argc);

    // check if user asked for -h
    if (has_option(args, "-h")) {
        print_help();
        return true;
    }

    // check if user asked for detection
    detection = has_option(args, "-d");
    if (detection) {
        string det;
        det = get_option(args, "-d");

        if (!is_number(det)) {
            return false;
        }

        show_img = std::stoi(det) > 0;
    }

    // check if user asked for segmentation
    segmentation = has_option(args, "-s");
    if (segmentation) {
        string seg;
        seg = get_option(args, "-s");

        if (!is_number(seg)) {
            return false;
        }

        show_steps = std::stoi(seg) > 0;
    }

    // check if user provided path to image
    if (has_option(args, "-image")) {
        path_to_image = get_option(args, "-image");
    }

    // check if user provided path to bounding boxes
    if (has_option(args, "-bb")) {
        path_to_bb = get_option(args, "-bb");
    }

    return detection || segmentation;
}

int main(int argc, char *argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    if (argc < 2) {
        cout << "Running detection and segmentation on test set\n";
        detection_demo();
        segmentation_demo();
        return 0;
    }

    bool detection = false;
    bool segmentation = false;
    bool show_image = false;
    bool show_steps = false;
    string path_to_image;
    string path_to_bb;

    if (!parse(argc, argv, detection, segmentation, show_image, show_steps, path_to_image, path_to_bb)) {
        print_help();
        return 0;
    }

    if (path_to_image.empty() || path_to_bb.empty()) {
        if (!path_to_image.empty()) {
            cout << "Ignoring path_to_image since path_to_bb has not been provided\n";
        }

        if (!path_to_bb.empty()) {
            cout << "Ignoring path_to_bb since path_to_image has not been provided\n";
        }

        if (detection) {
            cout << "Running detection on test set\n";
            detection_demo();
        }

        if (segmentation) {
            cout << "Running segmentation on test set\n";
            segmentation_demo();
        }
    }

    const cv::Mat img = cv::imread(path_to_image);
    vector<cv::Rect> bounding_boxes;
    vector<float> confidences;

    if (detection) {
        detect(img, bounding_boxes, confidences, show_image);
        detector::export_bb(bounding_boxes, path_to_bb);
    }

    if (segmentation) {
        cv::Mat output;
        segment(img, output, path_to_bb, show_steps);
    }
}