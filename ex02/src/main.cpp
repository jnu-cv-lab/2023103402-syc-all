#include <iostream>
#include <opencv2/opencv.hpp>
#include "func.h"
using namespace std;
using namespace cv;

int main() {
    Mat img = imread("test.jpg");  // 把你的图片改名为 test.jpg
    if (img.empty()) {
        cout << "图片不存在" << endl;
        return -1;
    }

    showInfo(img);        // 调用另一个文件的函数
    Mat gray = toGray(img);

    imshow("原图", img);
    imshow("灰度图", gray);
    imwrite("gray.jpg", gray);

    waitKey(0);
    return 0;
}