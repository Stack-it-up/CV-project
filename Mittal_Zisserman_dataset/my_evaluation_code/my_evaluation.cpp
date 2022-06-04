#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    for(int i=1; i<100; i++) {
        const string IMG_DIR = "../../training_dataset/training_data/images/";
        const string BBOX_DIR = "../../training_dataset/my_annotations/";
        const string FILENAME = "VOC2010_" + to_string(i);
        
        Mat result = imread(IMG_DIR+FILENAME+".jpg");
        //Vector<Rect> bboxes{};
        
        cout << IMG_DIR+FILENAME+".jpg\n";
        cout << BBOX_DIR+FILENAME+".mat.txt\n";
        
        ifstream input{BBOX_DIR + FILENAME + ".txt"};
        int x, y, width, height;
        
        while(input >> x >> y >> width >> height) {
            //bboxes.push_back(Rect{x,y,width, height)});
            cout << x << y << width << height << endl;
            rectangle(result, Rect{x,y,width, height}, Scalar{0,0,255}, 5);
        }
        
        input.close();
        
        imshow("img", result);
        waitKey(0);
    }
	
	return 0;
}
