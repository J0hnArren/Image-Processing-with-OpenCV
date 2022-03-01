#include <opencv2/opencv.hpp>
#include <string>


cv::Mat merge(const std::vector<cv::Mat>& img, const std::string& color) {
	cv::Mat final_image, g;
	g = cv::Mat::zeros(cv::Size(img[0].cols, img[0].rows), CV_8UC1);
	std::vector<cv::Mat> channels;
	if (color == "blue") {
		channels.push_back(img[0]);
		channels.push_back(g);
		channels.push_back(g);
		cv::merge(channels, final_image);
	} else if (color == "green") {
		channels.push_back(g);
		channels.push_back(img[1]);
		channels.push_back(g);
		cv::merge(channels, final_image);
	} else if (color == "red") {
		channels.push_back(g);
		channels.push_back(g);
		channels.push_back(img[2]);
		cv::merge(channels, final_image);
	}
	return final_image;
}

cv::Mat split_image(const cv::Mat& img) {
	std::vector<cv::Mat> rgb_channels(3);
	cv::split(img, rgb_channels);
	auto blue_img = merge(rgb_channels, "blue");
	auto green_img = merge(rgb_channels, "green");
	auto red_img = merge(rgb_channels, "red");
	cv::Mat up_row, down_row, result;
	cv::hconcat(img, red_img, up_row);
	cv::hconcat(green_img, blue_img, down_row);
	cv::vconcat(up_row, down_row, result);
	return result;
}

cv::Mat create_hist(cv::Mat img, const int& hist_w, const int& hist_h) {
	std::vector<cv::Mat> bgr_planes(3);
	split(img, bgr_planes);
	const int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange[] = { range };
	bool uniform = true, accumulate = false;
	int bin_w = cvRound((double)hist_w / histSize);

	cv::Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);

	cv::Mat hist_img(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::normalize(b_hist, b_hist, 0, hist_img.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(g_hist, g_hist, 0, hist_img.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(r_hist, r_hist, 0, hist_img.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 1; i < histSize; i++)
	{
		cv::line(hist_img, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);

		cv::line(hist_img, cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
			cv::Scalar(0, 255, 0), 2, 8, 0);

		cv::line(hist_img, cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	return hist_img;
}


int main() {
	const std::string image_path = "./data/cross_0256x0256.png";
	cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
	//cv::imshow("1", img);
	try {
		if (img.empty()) {
			throw "Could not read the image: ", image_path, '\n';
		}
	}
	catch (const cv::Exception& ex) {
		std::cout << ex.what() << '\n';
	}


	std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 25 };
	cv::imwrite("./cross_0256x0256_025.jpg", img, params);
	cv::Mat img_jpg = cv::imread("./cross_0256x0256_025.jpg");
	//cv::imshow("2", img_jpg);


	int hist_w = 1024, hist_h = 400;
	cv::Mat hist_image(hist_h, hist_w, CV_8UC3, 255);
	cv::Mat hist_image_jpg = hist_image.clone();

	// histogram
	cv::Mat hist = create_hist(img, hist_w, hist_h);
	cv::Mat hist_jpg = create_hist(img_jpg, hist_w, hist_h);

	cv::Mat sep(16, 1024, CV_8UC3);
	sep = cv::Scalar(255, 255, 255);
	cv::vconcat(hist, sep, hist);
	cv::vconcat(hist, hist_jpg, hist);
	cv::imwrite("./cross_0256x0256_hists.png", hist);


	//color
	auto img_chnl = split_image(img);
	auto cmpr_img_chnl = split_image(img_jpg);
	cv::imwrite("./cross_0256x0256_png_channels.png", img_chnl);
	cv::imwrite("./cross_0256x0256_jpg_channels.png", cmpr_img_chnl);

	imshow("Image", img );
	cv::waitKey(0);
	cv::imshow("Compressed image", img_jpg);
	cv::waitKey(0);
	cv::imshow("Histogram", hist);


	cv::waitKey(0);

	return EXIT_SUCCESS;
}