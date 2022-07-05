//
// Created by Giorgio on 04/06/22.
// Morphological operators tests
//
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

struct user_data{
    cv::Mat img;
    cv::Mat mask;
    int* iters_dil = nullptr;
    int* iters_ero = nullptr;
    int* t1 = nullptr;
    int* t2 = nullptr;
};

static void on_trackbar(int event, void* userdata) {

    user_data data = *(struct user_data*) userdata;

    int i_dil = *data.iters_dil;
    int i_ero = *data.iters_ero;
    cv::Mat img = data.img;
    cv::Mat mask = data.mask;
    int t1 = *data.t1;
    int t2 = *data.t2;

    cv::Mat canny, img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    cv::Canny(img_gray, canny, t1, t2);

    cv::Mat eroded;
    cv::Mat dilated;

    //standard kernel
    cv::Mat kernel = cv::Mat::zeros(3,3,CV_8UC1);
    kernel.at<unsigned char>(0,1) = 1;
    kernel.at<unsigned char>(1,0) = 1;
    kernel.at<unsigned char>(1,1) = 1;
    kernel.at<unsigned char>(1,2) = 1;
    kernel.at<unsigned char>(2,1) = 1;

    cv::erode(canny, eroded, kernel, cv::Point(-1,1), i_ero);
    cv::dilate(canny, dilated, kernel, cv::Point(-1,1), i_dil);
    cv::imshow("Erosion", eroded);
    cv::imshow("Dilation", dilated);

    cv::Mat open, close;
    cv::dilate(eroded, open, kernel, cv::Point(-1,1), i_dil);
    cv::erode(dilated, close, kernel, cv::Point(-1,1), i_ero);
    cv::imshow("Opening", open);
    cv::imshow("Closing", close);

    cv::Mat canny_mask, canny_close;

    cv::Canny(mask, canny_mask, t1, t2);
    cv::Canny(close, canny_close, t1, t2);

    cv::imshow("Canny on image", canny);
    cv::imshow("Canny on mask", canny_mask);
    cv::imshow("Canny on closing", canny_close);

}
int main() {

    cv::Mat img = cv::imread("../samples/23.jpg");

    /*
    cv::Mat channels[3];
    cv::split(img, channels);

    cv::equalizeHist(channels[0],channels[0]);
    cv::equalizeHist(channels[1],channels[1]);
    cv::equalizeHist(channels[2],channels[2]);

    cv::merge(channels, 3, img);
    */

    cv::Mat bgra, hsv;
    cv::cvtColor(img, bgra, cv::COLOR_BGR2BGRA);
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    cv::imshow("Image",img);

    //preprocessing to remove as much uninteresting parts as possible
    cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {

            cv::Vec4b pixel = bgra.at<cv::Vec4b>(i,j);
            cv::Vec3b hsv_pixel = hsv.at<cv::Vec3b>(i,j);

            //Thresholding based on https://arxiv.org/abs/1708.02694
            if(pixel[2] > 80 &&
               pixel[1] > 30 &&
               pixel[0] > 10 &&
               pixel[2] > pixel[1] &&
               pixel[2] > pixel[0] &&
               abs(pixel[2] - pixel[1]) > 10 &&
               pixel[3] > 10 &&
               hsv_pixel[0] >= 0 &&
               hsv_pixel[0] <= 60 &&
               hsv_pixel[1] >= 15 &&
               hsv_pixel[1] <= 80)
                mask.at<unsigned char>(i,j) = 255;
        }
    }

    cv::imshow("Mask", mask);

    int dil_iterations = 1;
    int ero_iterations = 1;
    int th1 = 1;
    int th2 = 1;

    std::string trackbar_ero = "Erosion iterations";
    std::string trackbar_dil = "Dilation iterations";
    std::string trackbar_t1 = "Threshold 1";
    std::string trackbar_t2 = "Threshold 2";

    cv::namedWindow("Erosion");
    cv::namedWindow("Dilation");
    cv::namedWindow("Opening");
    cv::namedWindow("Closing");
    cv::namedWindow("Canny on image");
    cv::namedWindow("Canny on mask");
    cv::namedWindow("Canny on closing");


    user_data data;
    data.img = img;
    data.mask = mask;
    data.iters_ero = &ero_iterations;
    data.iters_dil = &dil_iterations;
    data.t1 = &th1;
    data.t2 = &th2;

    cv::createTrackbar(trackbar_ero, "Erosion", &ero_iterations, 100, on_trackbar, (void*)&data);

    cv::createTrackbar(trackbar_dil, "Dilation", &dil_iterations, 100, on_trackbar, (void*)&data);

    cv::createTrackbar(trackbar_dil, "Opening", &dil_iterations, 100, on_trackbar, (void*)&data);
    cv::createTrackbar(trackbar_ero, "Opening", &ero_iterations, 100, on_trackbar, (void*)&data);

    cv::createTrackbar(trackbar_ero, "Closing", &ero_iterations, 100, on_trackbar, (void*)&data);
    cv::createTrackbar(trackbar_dil, "Closing", &dil_iterations, 100, on_trackbar, (void*)&data);

    cv::createTrackbar(trackbar_t1, "Canny on image", &th1, 300, on_trackbar, (void*)&data);
    cv::createTrackbar(trackbar_t2, "Canny on image", &th2, 300, on_trackbar, (void*)&data);

    cv::createTrackbar(trackbar_t1, "Canny on mask", &th1, 300, on_trackbar, (void*)&data);
    cv::createTrackbar(trackbar_t2, "Canny on mask", &th2, 300, on_trackbar, (void*)&data);

    cv::createTrackbar(trackbar_t1, "Canny on closing", &th1, 300, on_trackbar, (void*)&data);
    cv::createTrackbar(trackbar_t2, "Canny on closing", &th2, 300, on_trackbar, (void*)&data);

    cv::waitKey(0);
    return 0;
}