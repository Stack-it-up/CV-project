//
// Created by filippo on 07/07/22.
//
#include  <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include "Util.h"
#include "callbacks.h"

using namespace cv;
using namespace std;

int main() {
    string photos_dir = "../res/evaluation_data/rgb/*.jpg";
    string bboxes_dir = "../res/evaluation_data/det/*.txt";

    vector<string> photos_paths;
    vector<string> bbox_paths;
    glob(photos_dir, photos_paths);
    glob(bboxes_dir, bbox_paths);

    sort(photos_paths.begin(), photos_paths.end());
    sort(bbox_paths.begin(), bbox_paths.end());

    for(int i=20; i<photos_paths.size(); i++) {
        Mat input = imread(photos_paths[i]);

        const char* window_name = "TRACKBAAARS!!";
        namedWindow(window_name);
        /*
       int dim_gauss = 5;
       int dim = 15;
       int sigma_color = 1; //divided by 10
       int sigma_space = 1; //divided by 10

       Data d_gauss {vector<Mat*>{&input, &input},
                   vector<int*>{&dim_gauss},
                   window_name
       };
       createTrackbar( "dim:", window_name, &dim_gauss, 30, gaussian_threshold, &d_gauss );
       waitKey();

       Mat input_clone = input.clone();
       Data d_bil {vector<Mat*>{&input_clone, &input},
                   vector<int*>{&dim, &sigma_color, &sigma_space},
                   window_name
       };

       createTrackbar( "dim:", window_name, &dim, 50, bilateral_threshold, &d_bil );
       createTrackbar( "sigma_color:", window_name, &sigma_color, 1000, bilateral_threshold, &d_bil );
       createTrackbar( "sigma_space:", window_name, &sigma_space, 1000, bilateral_threshold, &d_bil );
       waitKey(0);
        */
        //bilateralFilter(input.clone(), input, 10, 80, 15);

        /*
         * //Luminance equalization, similar to histogram equalization for color images
         * NO NO BAD IDEA!
        Mat ycrcb{};
        cvtColor(input, ycrcb, COLOR_BGR2YCrCb);

        Mat channels[3];
        split(ycrcb, channels);
        equalizeHist(channels[0], channels[0]);
        merge(channels, 3, ycrcb); //for better equalization of color image!!

        imshow("", input);
        waitKey();
        cvtColor(ycrcb, input, COLOR_YCrCb2BGR);
        imshow("", input);
        waitKey();
         */
        Mat hsv{};
        cvtColor(input, hsv, COLOR_BGR2HSV_FULL);
        /*
        int sp = 1; //divided by 10
        int sr = 1; //divided by 10
        int maxlevel = 0;
        Mat ycrcb_clone = ycrcb.clone();

        Data d {vector<Mat*>{&ycrcb_clone, &ycrcb},
                      vector<int*>{&sp, &sr, &maxlevel},
                      window_name
        };

        createTrackbar( "spatial radius:", window_name, &sp, 500, meanshift_trackbar, &d );
        createTrackbar( "color radius:", window_name, &sr, 500, meanshift_trackbar, &d );
        createTrackbar( "max pyramid level:", window_name, &maxlevel, 8, meanshift_trackbar, &d );
        waitKey(0);
        */
        destroyAllWindows();

        pyrMeanShiftFiltering(hsv, hsv, 10, 17.5, 1);
        imshow("", hsv);
        waitKey();
        vector<Rect> boxes = extract_bboxes(bbox_paths[i], 10);

        vector<Mat> masks(boxes.size());
        for(int j=0; j<boxes.size(); j++) {
            Rect r = boxes[j];
            Mat bgmodel{};
            Mat fgmodel{};
            grabCut(hsv,
                    masks[j],
                    r,
                    bgmodel,
                    fgmodel,
                    12,
                    GC_INIT_WITH_RECT
            );
        }

        for(Mat mask : masks) {
            Mat output{};
            drawGrabcutMask(input, mask, output, 0.5);
            imshow("", output);
            waitKey(0);
        }

    }
}
