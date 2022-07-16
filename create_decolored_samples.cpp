#include <string>
#include <vector>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

int main() {
   string directory = "img";
   vector<string> pics;
   glob(directory, pics);

   for(string s : pics) {
      Mat img = imread(s, IMREAD_COLOR);
      cvtColor(img, img, COLOR_BGR2HSV);

      uchar hue = uchar(theRNG());

      for(int row=0; row<img.rows; row++) {
         for(int col=0; col<img.cols; col++) {
            Vec3b& x = img.at<Vec3b>(row, col);
            x[0] = hue;
         }
      }
      imwrite(s, img);
   }

   return 0;
}
