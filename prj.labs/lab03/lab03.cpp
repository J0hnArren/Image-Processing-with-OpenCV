#include <opencv2/opencv.hpp>
#include <cmath>


double intensity(const ptrdiff_t& i) {
    // tanh * sigmoid * const
    return tanh(i / 256.0) * 1 / (1 + exp(-i / 256.0)) * 256.0;
}

cv::Mat vividness() {
    uchar table[256];
    cv::Mat lookUpTable(1, 256, CV_8U);
    for (ptrdiff_t i = 0; i < 256; ++i) {
        lookUpTable.at<uchar>(0, i) = intensity(i);
    }
    return lookUpTable;
}

cv::Mat visualisation() {
    cv::Mat plot(512, 512, CV_8UC1, 255);
    line(plot, cv::Point(0, 511), cv::Point(0, 0), cv::Scalar(128, 128, 128), 2);
    line(plot, cv::Point(511, 0), cv::Point(0, 0), cv::Scalar(128, 128, 128), 2);
    for (ptrdiff_t i = 0; i < plot.cols; ++i) {
        circle(plot, cv::Point(i, 511 - intensity(i)), 1, cv::Scalar(0), -1);
    }
    return plot;
}

int main() {
	const std::string image_path = "./data/cross_0256x0256.png";

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    cv::Mat img_gray, img_res, img_gray_res;
    cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    LUT(img, vividness(), img_res);
    LUT(img_gray, vividness(), img_gray_res);
    imwrite("lab03_rgb.png", img);
    imwrite("lab03_gre.png", img_gray);
    imwrite("lab03_rgb_res.png", img_res);
    imwrite("lab03_gre_res.png", img_gray_res);

    // plot
    imwrite("lab03_viz_func.png", visualisation());

	cv::waitKey(0);

	return EXIT_SUCCESS;
}