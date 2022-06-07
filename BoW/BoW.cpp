#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void computeFeaturesClusters(vector<Mat>& train_images, vector<Mat>& train_masks, Mat& dictionary);
void computeSift(Mat& img, vector<KeyPoint>& keypoints, Mat& descriptors);
void loadImages(vector<Mat>& images, string& folder_path);
void loadBoundingBox(vector<vector<int>>& bounding_boxes, string& folder_path);

int main(int argc, char** argv) {
	string train_images_path = "../../Dataset progetto CV - Hand detection _ segmentation/rgb/*.jpg";
	string train_masks_path = "../../Dataset progetto CV - Hand detection _ segmentation/mask/*.png";
	string train_bounds_path = "../../Dataset progetto CV - Hand detection _ segmentation/det/*.txt";
	vector<Mat> train_images;
	vector<Mat> train_masks;
	vector<Mat> train_segmented;
	Mat dictionary;

	// Load images
	loadImages(train_images, train_images_path);
	loadImages(train_masks, train_masks_path);

	computeFeaturesClusters(train_images, train_masks, dictionary);
	FileStorage fs_read("dictionary.yml", FileStorage::READ);
	fs_read["vocabulary"] >> dictionary;
	fs_read.release();

	// Second part
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher());
	Ptr<FeatureDetector> detector = SIFT::create();
	Ptr<DescriptorExtractor> extractor = SIFT::create();
	BOWImgDescriptorExtractor bow_de(extractor, matcher);
	bow_de.setVocabulary(dictionary);

	FileStorage fs("descriptor.yml", FileStorage::WRITE);

	for (Mat img : train_images) {
		vector<KeyPoint> keypoints;
		Mat bow_descriptor;
		detector->detect(img, keypoints);
		bow_de.compute(img, keypoints, bow_descriptor);

		Mat keypoints_image;
		cv::drawKeypoints(img, keypoints, keypoints_image);
		imshow("keypoints", keypoints_image);

		waitKey(0);

		fs << "img" << bow_descriptor;
	}

	return 0;
}

void computeFeaturesClusters(vector<Mat>& train_images, vector<Mat>& train_masks, Mat& dictionary) {
	vector<Mat> train_segmented;
	Mat features;

	for (int i = 0; i < train_images.size(); i++) {
		cv::threshold(train_masks[i], train_masks[i], 200, 255, THRESH_BINARY);
		Mat image_segmented;
		cv::bitwise_and(train_images[i], train_masks[i], image_segmented);
		train_segmented.push_back(image_segmented);

		Mat descriptor;
		vector<KeyPoint> keypoints;
		computeSift(image_segmented, keypoints, descriptor);
		features.push_back(descriptor);

		/*Mat image_keypoints;
		cv::drawKeypoints(train_images[i], keypoints, image_keypoints);
		imshow("keypoints", image_keypoints);
		waitKey(0);*/
	}

	int dictionary_size = 10;
	TermCriteria tc(TermCriteria::MAX_ITER, 100, 0.001);
	
	BOWKMeansTrainer bowTrainer(dictionary_size, tc, 1, KMEANS_PP_CENTERS);
	dictionary = bowTrainer.cluster(features);

	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
}

void computeSift(Mat& img, vector<KeyPoint>& keypoints, Mat& descriptors) {
	Ptr<SIFT> detector = SIFT::create(20);
	detector->detectAndCompute(img, noArray(), keypoints, descriptors);
}

void loadImages(vector<Mat>& images, string& folder_path) {
	vector<cv::String> img_names;
	glob(folder_path, img_names, false);

	for (size_t i = 0; i < img_names.size(); i++) {
		images.push_back(imread(img_names[i]));
	}
}


void loadBoundingBox(vector<vector<int>>& bounding_boxes, string& folder_path) {
	vector<cv::String> file_names;
	glob(folder_path, file_names, false);

	for (size_t i = 0; i < file_names.size(); i++) {

	}
}