#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/shape.hpp>
#include "highgui.h"
#include <time.h>
#include <math.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <vector>
#include <fstream>
#include <thread>

using namespace std;
using namespace dlib;
using namespace cv;

// method_0508_1
void get_image_warping(Mat face_show, Mat sunglasses_src, Mat &sunglasses_dst, Mat sunglasses_data, std::vector<full_object_detection> shapes, int index, Mat &flag_warp, bool keep_shape)
{
	//初始化关键结点
	std::vector<Point2f> srcPoints, dstPoints;

	/*----src_points----*/
	srcPoints.emplace_back(sunglasses_data.at<float>(0, 3), sunglasses_data.at<float>(1, 3));
	srcPoints.emplace_back(sunglasses_data.at<float>(0, 6), sunglasses_data.at<float>(1, 6));
	srcPoints.emplace_back(sunglasses_data.at<float>(0, 2), sunglasses_data.at<float>(1, 2));
	srcPoints.emplace_back(sunglasses_data.at<float>(0, 4), sunglasses_data.at<float>(1, 4));
	srcPoints.emplace_back(sunglasses_data.at<float>(0, 7), sunglasses_data.at<float>(1, 7));
	srcPoints.emplace_back(sunglasses_data.at<float>(0, 11), sunglasses_data.at<float>(1, 11));
	srcPoints.emplace_back(sunglasses_data.at<float>(0, 1), sunglasses_data.at<float>(1, 1));
	srcPoints.emplace_back(sunglasses_data.at<float>(0, 8), sunglasses_data.at<float>(1, 8));
	srcPoints.emplace_back(sunglasses_data.at<float>(0, 0), sunglasses_data.at<float>(1, 0));
	srcPoints.emplace_back(sunglasses_data.at<float>(0, 9), sunglasses_data.at<float>(1, 9));

	/*----dst_points----*/
	float x_1 = 0.0, x_2 = 0.0, x_3 = 0.0, x_4 = 0.0;
	float y_1 = 0.0, y_2 = 0.0, y_3 = 0.0, y_4 = 0.0;
	float dst_x = 0.0, dst_y = 0.0;

	//dst_0
	x_1 = shapes[index].part(39).x();
	y_1 = shapes[index].part(39).y();
	x_2 = shapes[index].part(27).x();
	y_2 = shapes[index].part(27).y();
	dst_x = (x_1 + x_2) / 2.0;
	dst_y = (y_1 + y_2) / 2.0;
	dstPoints.emplace_back(dst_x, dst_y);

	//dst_1
	x_1 = shapes[index].part(27).x();
	y_1 = shapes[index].part(27).y();
	x_2 = shapes[index].part(42).x();
	y_2 = shapes[index].part(42).y();
	dst_x = (x_1 + x_2) / 2.0;
	dst_y = (y_1 + y_2) / 2.0;
	dstPoints.emplace_back(dst_x, dst_y);

	//dst_2
	x_1 = shapes[index].part(19).x();
	y_1 = shapes[index].part(19).y();
	x_2 = shapes[index].part(21).x();
	y_2 = shapes[index].part(21).y();
	x_3 = shapes[index].part(38).x();
	y_3 = shapes[index].part(38).y();
	x_4 = shapes[index].part(39).x();
	y_4 = shapes[index].part(39).y();
	dst_x = (x_1 + x_2 + x_3 + x_4) / 4.0;
	dst_y = (y_1 + y_2 + y_3 + y_4) / 4.0;
	dstPoints.emplace_back(dst_x, dst_y);

	//dst_3
	float dst_x_ = 0.0, dst_y_ = 0.0;
	dst_x_ = (shapes[index].part(40).x() + shapes[index].part(29).x()) / 2;
	dst_y_ = (shapes[index].part(40).y() + shapes[index].part(29).y()) / 2;
	dstPoints.emplace_back(dst_x_, dst_y_);
	//circle(face_show, cvPoint(dst_x_, dst_y_), 2, cvScalar(0, 0, 255), -1);

	//dst_4
	x_1 = shapes[index].part(22).x();y_1 = shapes[index].part(22).y();
	x_2 = shapes[index].part(24).x();y_2 = shapes[index].part(24).y();
	x_3 = shapes[index].part(42).x();y_3 = shapes[index].part(42).y();
	x_4 = shapes[index].part(43).x();y_4 = shapes[index].part(43).y();
	dst_x = (x_1 + x_2 + x_3 + x_4) / 4.0;
	dst_y = (y_1 + y_2 + y_3 + y_4) / 4.0;
	dstPoints.emplace_back(dst_x, dst_y);

	//dst_5
	dst_x_ = 0.0, dst_y_ = 0.0;
	dst_x_ = (shapes[index].part(47).x() + shapes[index].part(29).x()) / 2;
	dst_y_ = (shapes[index].part(47).y() + shapes[index].part(29).y()) / 2;
	dstPoints.emplace_back(dst_x_, dst_y_);
	//circle(face_show, cvPoint(dst_x_, dst_y_), 2, cvScalar(0, 0, 255), -1);

	//dst_6
	x_1 = shapes[index].part(17).x();
	y_1 = shapes[index].part(17).y();
	x_2 = shapes[index].part(19).x();
	y_2 = shapes[index].part(19).y();
	x_3 = shapes[index].part(36).x();
	y_3 = shapes[index].part(36).y();
	x_4 = shapes[index].part(37).x();
	y_4 = shapes[index].part(37).y();
	dst_x = (x_1 + x_2 + x_3 + x_4) / 4.0;
	dst_y = (y_1 + y_2 + y_3 + y_4) / 4.0;
	dstPoints.emplace_back(dst_x, dst_y);
	//circle(face_show, cvPoint(dst_x, dst_y), 2, cvScalar(0, 0, 255), -1);

	//dst_8
	x_1 = shapes[index].part(24).x();
	y_1 = shapes[index].part(24).y();
	x_2 = shapes[index].part(26).x();
	y_2 = shapes[index].part(26).y();
	x_3 = shapes[index].part(44).x();
	y_3 = shapes[index].part(44).y();
	x_4 = shapes[index].part(45).x();
	y_4 = shapes[index].part(45).y();
	dst_x = (x_1 + x_2 + x_3 + x_4) / 4.0;
	dst_y = (y_1 + y_2 + y_3 + y_4) / 4.0;
	dstPoints.emplace_back(dst_x, dst_y);

	//dst_10
	x_1 = shapes[index].part(39).x();
	y_1 = shapes[index].part(39).y();
	x_2 = shapes[index].part(27).x();
	y_2 = shapes[index].part(27).y();
	x_3 = shapes[index].part(36).x();
	y_3 = shapes[index].part(36).y();
	dst_x = x_3 - (x_2 - x_1) / 3.0 * 2;
	dst_y = y_3 - (y_2 - y_1) / 3.0 * 2;
	dstPoints.emplace_back(dst_x, dst_y);

	//dst_11
	x_1 = shapes[index].part(27).x();
	y_1 = shapes[index].part(27).y();
	x_2 = shapes[index].part(42).x();
	y_2 = shapes[index].part(42).y();
	x_3 = shapes[index].part(45).x();
	y_3 = shapes[index].part(45).y();
	dst_x = x_3 + (x_2 - x_1) / 3.0 * 2;
	dst_y = y_3 + (y_2 - y_1) / 3.0 * 2;
	dstPoints.emplace_back(dst_x, dst_y);

	//img_warp
	int max_h = std::max(face_show.rows, sunglasses_src.rows);
	int max_w = std::max(face_show.cols, sunglasses_src.cols);
	Mat temp = Mat::zeros(max_h, max_w, CV_8UC3);
	Rect roi(0, 0, sunglasses_src.cols, sunglasses_src.rows);
	sunglasses_src.copyTo(temp(roi));

	std::vector<DMatch> match;
	for (int i = 0; i < dstPoints.size(); i++)
	{
		match.push_back(DMatch(i, i, 0));
	}

	if (keep_shape)
	{
		dstPoints.emplace_back(0, 0);
		dstPoints.emplace_back(0, face_show.rows - 1);
		dstPoints.emplace_back(face_show.cols - 1, 0);
		dstPoints.emplace_back(face_show.cols - 1, face_show.rows - 1);

		srcPoints.emplace_back(0, 0);
		srcPoints.emplace_back(0, sunglasses_src.rows - 1);
		srcPoints.emplace_back(sunglasses_src.cols - 1, 0);
		srcPoints.emplace_back(sunglasses_src.cols - 1, sunglasses_src.rows - 1);

		for (int i = dstPoints.size(); i < dstPoints.size() + 4; i++)
		{
			match.emplace_back(i, i, 0);
		}
	}

	// Ptr<AffineTransformer> tps = createAffineTransformer(false);
	Ptr<ThinPlateSplineShapeTransformer> tps = createThinPlateSplineShapeTransformer(0);
	tps->estimateTransformation(dstPoints, srcPoints, match);
	Mat temp2;
	tps->warpImage(temp, temp2);

	Mat output(sunglasses_src.size(), CV_8UC3);
	Rect roi_2(0, 0, face_show.cols, face_show.rows);
	temp2(roi_2).copyTo(output);
	sunglasses_dst = output.clone();

	//warp flag
	Mat flag(sunglasses_src.size(), CV_8UC3);
	for (int i_ = 0; i_ < flag.rows; i_++)
	{
		for (int j_ = 0; j_ < flag.cols; j_++)
		{
			flag.at<Vec3b>(i_, j_)[0] = 255;
			flag.at<Vec3b>(i_, j_)[1] = 255;
			flag.at<Vec3b>(i_, j_)[2] = 255;
		}
	}

	flag.copyTo(temp(roi));
	tps->warpImage(temp, temp2);
	temp2(roi_2).copyTo(output);
	flag_warp = output.clone();
}

//Guided Filter
Mat guidedfilter(cv::Mat I, cv::Mat p, int r, double eps)
{
	/*
	% GUIDEDFILTER   O(1) time implementation of guided filter.
	%
	%   - guidance image: I (should be a gray-scale/single channel image)
	%   - filtering input image: p (should be a gray-scale/single channel image)
	%   - local window radius: r
	%   - regularization parameter: eps
	*/

	cv::Mat _I;
	I.convertTo(_I, CV_64FC1);
	I = _I;

	cv::Mat _p;
	p.convertTo(_p, CV_64FC1);
	p = _p;

	//[hei, wid] = size(I);
	int hei = I.rows;
	int wid = I.cols;

	//N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.
	cv::Mat N;
	cv::boxFilter(cv::Mat::ones(hei, wid, I.type()), N, CV_64FC1, cv::Size(r, r));

	//mean_I = boxfilter(I, r) ./ N;
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, CV_64FC1, cv::Size(r, r));

	//mean_p = boxfilter(p, r) ./ N;
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));

	//mean_Ip = boxfilter(I.*p, r) ./ N;
	cv::Mat mean_Ip;
	cv::boxFilter(I.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;
	cv::Mat mean_II;
	cv::boxFilter(I.mul(I), mean_II, CV_64FC1, cv::Size(r, r));

	//var_I = mean_II - mean_I .* mean_I;
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;
	cv::Mat a = cov_Ip / (var_I + eps);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
	cv::Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;
	cv::Mat mean_a;
	cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
	//mean_a = mean_a / N;
	mean_a = mean_a;
	//mean_b = boxfilter(b, r) ./ N;
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
	//mean_b = mean_b / N;
	mean_b = mean_b;
	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
	cv::Mat q = mean_a.mul(I) + mean_b;

	return q;
}

//Get image
Mat getimage(Mat &a)
{
	int hei = a.rows;
	int wid = a.cols;
	Mat I(hei, wid, CV_64FC1);
	//convert image depth to CV_64F
	a.convertTo(I, CV_64FC1, 1.0 / 255.0);
	return I;
}

//Guided Filter interface
void guidedFilter(Mat guide, Mat src, Mat &dst, int r, double eps)
{

	std::vector<Mat> guide_bgr_src, guide_bgr_dst;
	std::vector<Mat> src_bgr_src, src_bgr_dst;

	split(guide, guide_bgr_src);
	split(src, src_bgr_src);

	Mat dst_guide;

	for (int i = 0; i < 3; i++)
	{
		Mat I = getimage(guide_bgr_src[i]);
		Mat p = getimage(src_bgr_src[i]);
		Mat q = guidedfilter(I, p, r, eps);
		src_bgr_dst.push_back(q);
	}
	merge(src_bgr_dst, dst_guide);
	dst_guide = dst_guide * 255;
	dst_guide.convertTo(dst, CV_8UC3);
	dst_guide.release();
}

void process(const int s_num)
{

	//路径设置
	string sunglasses_path = "/home/xurui/sunglasses_temp_task/sunglasses/";
	string annotation_path = "/home/xurui/sunglasses_temp_task/sunglasses_annotation/";
	string file_type = ".jpg";

	//加载人脸检测和姿态估计模型
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

	//读取墨镜图片
	Mat sunglasses_img = imread(sunglasses_path + to_string(s_num) + file_type);
	Mat sunglasses_warp;

	//读取墨镜标注数据
	char s_1[100], s_2[100];
	sprintf(s_1, "/home/xurui/sunglasses_temp_task/sunglasses_annotation/%d.xml", s_num);
	sprintf(s_2, "sunglasses");
	FileStorage fs(s_1, FileStorage::READ);
	Mat sunglasses_data;
	fs[s_2] >> sunglasses_data;

	//读取人脸图片地址
	ifstream myfile("/home/xurui/sunglasses_temp_task/name_list.txt");
	//ofstream outfile("F:/temp_task/batch_process_test/batch_process_test/output_%d.txt", s_num, ios::app);
	string face_path;

	if (!myfile.is_open())
	{
		cout << "未成功打开图片地址文件" << endl;
	}
	while (getline(myfile, face_path))
	{
		if (s_num % 25 == 0)
		{
			cout << s_num << "  " << face_path << endl;
		}

		//读取人脸图片
		Mat face_img = imread(face_path);
		Mat face_show = face_img.clone();

		//使用dlib进行人脸特征点检测
		cv_image<bgr_pixel> cimg(face_img);
		std::vector<dlib::rectangle> faces = detector(cimg);
		std::vector<full_object_detection> shapes;
		for (unsigned long i = 0; i < faces.size(); ++i)
		{
			shapes.push_back(pose_model(cimg, faces[i]));
		}

		//if (!shapes.empty()) {
		//	for (int j = 0; j < shapes.size(); j++) {
		//		for (int i = 0; i < 68; i++) {
		//			circle(face_show, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 2, cvScalar(255, 0, 0), -1);
		//		}
		//	}
		//}

		if (!shapes.empty())
		{
			for (int j = 0; j < shapes.size(); j++)
			{
				//设置flag
				Mat flag_warp;
				//warp操作
				get_image_warping(face_show, sunglasses_img, sunglasses_warp, sunglasses_data, shapes, j, flag_warp, 1);

				//贴墨镜的mask
				for (int i_ = 0; i_ < sunglasses_warp.rows; i_++)
				{
					for (int j_ = 0; j_ < sunglasses_warp.cols; j_++)
					{
						if (flag_warp.at<Vec3b>(i_, j_)[0] == 255)
						{
							if (sunglasses_warp.at<Vec3b>(i_, j_)[0] >= 240 && sunglasses_warp.at<Vec3b>(i_, j_)[1] >= 240 && sunglasses_warp.at<Vec3b>(i_, j_)[2] >= 240)
							{
								continue;
							}
							else
							{
								flag_warp.at<Vec3b>(i_, j_)[0] = 0;
							}
						}
					}
				}

				//获取自定义核
				Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
				//腐蚀操作
				erode(sunglasses_warp, sunglasses_warp, element);

				//贴墨镜
				for (int i_ = 0; i_ < sunglasses_warp.rows; i_++)
				{
					for (int j_ = 0; j_ < sunglasses_warp.cols; j_++)
					{
						if (flag_warp.at<Vec3b>(i_, j_)[0] < 200 && flag_warp.at<Vec3b>(i_, j_)[1] >= 240 && flag_warp.at<Vec3b>(i_, j_)[2] >= 240)
						{
							face_show.at<Vec3b>(i_, j_)[0] = sunglasses_warp.at<Vec3b>(i_, j_)[0];
							face_show.at<Vec3b>(i_, j_)[1] = sunglasses_warp.at<Vec3b>(i_, j_)[1];
							face_show.at<Vec3b>(i_, j_)[2] = sunglasses_warp.at<Vec3b>(i_, j_)[2];
						}
					}
				}
				//Guided Filter
				guidedFilter(face_img, face_show, face_show, 3, 0.0001);
			}
			//----保存图片----
			//如果当前文件夹不存在则创建新的文件夹
			string temp_path = "/home/xurui/sunglasses_output/";
			string temp_dir = temp_path.append(to_string(s_num)).append("/CASIA-WebFace/").append(face_path.substr(26, 7));
			string save_path = temp_dir.append("/").append(face_path.substr(34, 7));
			imwrite(save_path, face_show);
		}

		//outfile << temp;
		//outfile << endl;
	}

	myfile.close();

	//释放内存
	sunglasses_img.release();
	sunglasses_data.release();
	fs.release();
}

int main(int argc, char *argv[])
{

	//多线程处理
	thread *task[30];
	for (int i = 30; i < 51; i++)
		task[i] = new thread(process, i + 1);
	for (int i = 30; i < 51; i++)
		task[i]->join();

	return 0;
}