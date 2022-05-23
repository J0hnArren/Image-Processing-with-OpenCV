#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>


const int rows = 2, cols = 3, SIZE = 150;
cv::Mat filter_1, filter_2;
cv::Mat img(rows* SIZE, cols* SIZE, CV_32FC1);
cv::Rect2i rect(0, 0, SIZE, SIZE);

const cv::Mat conv_1(3, 3, CV_32FC1, new float[9]{ 1, 0, -1, 2, 0, -2, 1, 0, -1 });
const cv::Mat conv_2(3, 3, CV_32FC1, new float[9]{ 1, 2, 1, 0, 0, 0, -1, -2, -1 });
const std::vector<int> colours = { 0, 127, 255, 127, 255, 0 };


void draw_img() {
	for (int x = 0; x < rows; ++x) {
		for (int y = 0; y < cols; ++y) {
			cv::Mat cell(SIZE, SIZE, CV_32FC1);
			cell = colours[x * cols + y];
			int circleColor = colours[(x * cols + y + cols) % (rows * cols)];
			int radius = SIZE / 2;
			cv::Point circleCenter(radius, radius);
			cv::circle(cell, circleCenter, radius, circleColor, cv::FILLED);
			cell.copyTo(img(rect));
			rect.x = (y + 1) * rect.width;
		}
		rect.x = 0;
		rect.y = (x + 1) * rect.height;
	}
}

cv::Mat avg_geom(const cv::Mat& f1, const cv::Mat& f2) {
	cv::Mat result(f1);
	result = 0;

	for (int x = 0; x < result.cols; ++x) {
		for (int y = 0; y < result.rows; ++y) {
			auto geom_avg = sqrt(pow(f1.at<float>(y, x), 2) + pow(f2.at<float>(y, x), 2));
			if (geom_avg < 0)
				result.at<float>(y, x) = (geom_avg + 255) / 2;
			else
				result.at<float>(y, x) = geom_avg;
		}
	}

	return result;
}


int main() {
	draw_img();

	cv::filter2D(img, filter_1, -1, conv_1);
	cv::filter2D(img, filter_2, -1, conv_2);

	cv::imwrite("./output/drawed.png", img);
	cv::imwrite("./output/filter_1.png", filter_1);
	cv::imwrite("./output/filter_2.png", filter_2);
	cv::imwrite("./output/avg_geom.png", avg_geom(filter_1, filter_2));

	cv::waitKey(0);
}