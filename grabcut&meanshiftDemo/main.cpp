#include  <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

/*
 * Draw grabcut mask with different color and 30% transparency
 * */

void drawGrabcutMask(Mat& image, Mat& mask, Mat& output) {
    Scalar FG_COLOR{255, 0, 0};
    Scalar PROB_FG_COLOR{0,150,0};

    Scalar BG_COLOR{0, 0, 255};
    Scalar PROB_BG_COLOR{0,150,150};

    Mat colored_mask{mask.size(), CV_8UC3};

    colored_mask.setTo(BG_COLOR, mask==0);
    colored_mask.setTo(FG_COLOR, mask==1);
    colored_mask.setTo(PROB_BG_COLOR, mask==2);
    colored_mask.setTo(PROB_FG_COLOR, mask==3);

    output.create(image.size(), CV_8UC3);
    addWeighted(image, 0, colored_mask, 1, 0, output);
}

int main() {
    string photos_dir = "/home/filippo/Desktop/unipd/4anno/computer_vision/final_project/datasets/Dataset progetto CV - Hand detection _ segmentation/rgb/*.jpg";
    string bboxes_dir = "/home/filippo/Desktop/unipd/4anno/computer_vision/final_project/datasets/Dataset progetto CV - Hand detection _ segmentation/det/*.txt";

    vector<string> photos_paths;
    vector<string> bbox_paths;
    glob(photos_dir, photos_paths);
    glob(bboxes_dir, bbox_paths);

    sort(photos_paths.begin(), photos_paths.end());
    sort(bbox_paths.begin(), bbox_paths.end());

    for(int i=12; i<photos_paths.size(); i++) {
        Mat input = imread(photos_paths[i]);
        //GaussianBlur(input, input, Size(5,5), 0, 0);
        //bilateralFilter(input.clone(), input, 5, 2, 8);
        imshow("", input);
        waitKey(0);

        ifstream boxes_txt;
        boxes_txt.open(bbox_paths[i]);

        vector<Rect> boxes;
        while(!boxes_txt.eof()) {
            int x, y, w, h; //params of the rectangle to be read from file
            boxes_txt >> x >> y >> w >> h;
            //boxes.push_back(Rect{x-30, y-30, w+60, h+60}); //draw a 30 pixel border around the hand
            boxes.push_back(Rect{x, y, w, h});
        }
        boxes_txt.close();

        vector<Mat> masks(boxes.size());
        for(int j=0; j<boxes.size(); j++) {
            Rect r = boxes[j];
            rectangle(input, r, Scalar{0, 0, 255});
            Mat bgmodel{};
            Mat fgmodel{};
            grabCut(input,
                    masks[j],
                    r,
                    bgmodel,
                    fgmodel,
                    15,
                    GC_INIT_WITH_RECT
            );
        }
        imshow("", input);
        waitKey(0);

        for(Mat mask : masks) {
            Mat output{};
            drawGrabcutMask(input, mask, output);
            imshow("", output);
            waitKey(0);
        }
    }
    return 0;
}
