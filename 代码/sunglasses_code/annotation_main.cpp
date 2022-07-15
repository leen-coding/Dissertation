#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <opencv2\opencv.hpp>  
#include "highgui.h"
#include <time.h>
#include <math.h>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

	//路径设置
	string path = "sunglasses/";
	string file_type = ".jpg";
	
	//读取图片
	int num = 6;
	Mat img = imread(path + to_string(num) + file_type);

	//手动标注数据
	Mat data(2, 12, CV_32F);

	//面向图片的左边镜片(x,y)
	data.at<float>(0, 0) = 99;
	data.at<float>(1, 0) = 282;

	data.at<float>(0, 1) = 140;
	data.at<float>(1, 1) = 233;

	data.at<float>(0, 2) = 211; 
	data.at<float>(1, 2) = 219;

	data.at<float>(0, 3) = 274; 
	data.at<float>(1, 3) = 260;

	data.at<float>(0, 4) = 228;
	data.at<float>(1, 4) = 333;

	data.at<float>(0, 5) = 108;
	data.at<float>(1, 5) = 344;
	
	//面向图片的右边镜片(x,y)
	data.at<float>(0, 6) = 326;
	data.at<float>(1, 6) = 259;
	
	data.at<float>(0, 7) = 376;
	data.at<float>(1, 7) = 220;
	
	data.at<float>(0, 8) = 453;
	data.at<float>(1, 8) = 227;
	
	data.at<float>(0, 9) = 503;
	data.at<float>(1, 9) = 276;
	
	data.at<float>(0, 10) = 500;
	data.at<float>(1, 10) = 339;
	
	data.at<float>(0, 11) = 374;
	data.at<float>(1, 11) = 328;
	
	//测试
	for (int i = 0; i < 12; i++) {
		circle(img, cvPoint(data.at<float>(0, i), data.at<float>(1, i)), 2, cvScalar(255, 0, 0), -1);
	}

	char s_1[100], s_2[100];
	sprintf_s(s_1, "sunglasses_annotation/%d.xml", num);
	sprintf_s(s_2, "sunglasses");
	cv::FileStorage fs(s_1, cv::FileStorage::WRITE);
	fs << s_2 << data;
	fs.release();

	cout << "finish" << endl;
	cout << "end" << endl;
}