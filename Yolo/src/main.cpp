//
// Created by giorgio on 12/07/22.
//
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;

int main() {

    float CONF_THRESH, NMS_THRESH = 0.5;

    String cfg = "../cfg/yolov3-custom.cfg";
    String weights = "../cfg/TO_COMPLETE"; //Put the weights file under cfg directory

    dnn::Net net = dnn::readNetFromDarknet(cfg, weights);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);

    std::vector<String> layers = net.getLayerNames();
    std::vector<int> indexes = net.getUnconnectedOutLayers();
    std::vector<String> output_layers;

    for(int i = 0; i < std::size(indexes); i++) {

        int index = indexes[i];
        output_layers.push_back(layers[index - 1]);
    }

    Mat img = imread("../samples/image_2.jpg");

    int height = img.rows;
    int width = img.cols;

    double scale = 0.00392;
    Size sz = Size(416,416);
    Mat blob = dnn::blobFromImage(img, scale, sz, Scalar(), true);
    net.setInput(blob);

    std::vector<Mat> layer_outputs;

    for(int i = 0; i < size(output_layers); i++) {

        layer_outputs.push_back(net.forward(output_layers[i]));
    }

    std::vector<Rect> bounding_boxes;
    std::vector<float> confidences;

    for(int i = 0; i < size(layer_outputs); i++) {

        Mat output = layer_outputs.at(i);

        for(int j = 0; j < output.rows; j++) {

            float confidence = output.at<float>(j,5);

            if(confidence > CONF_THRESH) {

                float x_center = output.at<float>(j,0) * width;
                float y_center = output.at<float>(j,1) * height;
                float w = output.at<float>(j,2) * width;
                float h = output.at<float>(j,3) * height;

                float x = (x_center - w / 2);
                float y = (y_center - h / 2);

                Rect bounding_box = Rect(x, y,w,h);

                bounding_boxes.push_back(bounding_box);
                confidences.push_back(confidence);
            }

        }
    }

    std::vector<int> bbox_indexes;

    dnn::NMSBoxes(bounding_boxes, confidences, CONF_THRESH, NMS_THRESH, bbox_indexes);

    for(int i = 0; i < size(bbox_indexes); i++) {
        int index = bbox_indexes.at(i);
        Rect bounding_box = bounding_boxes.at(index);
        Scalar color = Scalar(0,255,0);
        rectangle(img, bounding_box, color, 2);
    }

    imshow("Detection", img);
    waitKey(0);
    return 0;
}