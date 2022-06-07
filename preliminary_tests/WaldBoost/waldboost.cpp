//
// Created by filippo on 06/06/22.
//

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xobjdetect.hpp>


using namespace cv;
using namespace std;

int main() {
    const string POS_FOLDER = "data/training_positive_small/";
    const string NEG_FOLDER = "data/training_negative_small";
    const string TEST_FOLDER = "data/rgb";
    Ptr<xobjdetect::WBDetector> detector = xobjdetect::WBDetector::create();

    //training
    detector->train(POS_FOLDER, NEG_FOLDER);
    FileStorage trained_model = FileStorage("model/model.pretrained", FileStorage::FORMAT_AUTO + FileStorage::WRITE);
    detector->write(trained_model);

    //test
    //TODO
    return 0;
}



