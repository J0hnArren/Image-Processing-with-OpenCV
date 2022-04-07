#include <opencv2/opencv.hpp>
#include <cmath>


double intensity(const ptrdiff_t& i) {
    // tanh * sigmoid * const
    return tanh(i / 256.0) * 1 / (1 + exp(-i / 256.0)) * 256.0;
}

cv::Mat vividness() {
    cv::Mat lookUpTable(1, 256, CV_8U);
    for (ptrdiff_t i = 0; i < 256; ++i) {
        lookUpTable.at<uchar>(0, i) = intensity(i);
    }
    return lookUpTable;
}

cv::Mat visualisation(const cv::Mat &lup_table) {
    const int bottom = 512 / 2 - 512 / 2;
    const int top = 512 / 2 + 512 / 2 - 1;
    cv::Mat plot(512, 512, CV_8UC3);
    plot = cv::Scalar(255, 255, 255);
    cv::line(plot, cv::Point2i(bottom, bottom), cv::Point2i(bottom, top), cv::Scalar(255, 255, 0), 2);
    cv::line(plot, cv::Point2i(bottom, top), cv::Point2i(top, top), cv::Scalar(255, 255, 0), 2);
    for (int i = 0; i < 256; ++i) {
        const int x = bottom + static_cast<int>((i / 256.0) * 512);
        const int y = top - static_cast<int>((lup_table.at<uchar>(i) / 256.0) * 512);
        plot.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
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
    const cv::Mat lup_table = vividness();

    // plot
    imwrite("lab03_viz_func.png", visualisation(lup_table));

    LUT(img, lup_table, img_res);
    LUT(img_gray, lup_table, img_gray_res);
    imwrite("lab03_rgb.png", img);
    imwrite("lab03_gre.png", img_gray);
    imwrite("lab03_rgb_res.png", img_res);
    imwrite("lab03_gre_res.png", img_gray_res);

	cv::waitKey(0);

	return EXIT_SUCCESS;
}