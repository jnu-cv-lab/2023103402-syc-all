#include "func.h"
#include <iostream>
using namespace std;
using namespace cv;

void showInfo(const Mat& img) {
    cout << "宽度: " << img.cols << endl;
    cout << "高度: " << img.rows << endl;
    cout << "通道: " << img.channels() << endl;
}

Mat toGray(const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    return gray;
}