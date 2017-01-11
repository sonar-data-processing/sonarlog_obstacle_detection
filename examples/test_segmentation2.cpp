#include <iostream>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "base/test_config.h"
#include "rock_util/LogReader.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonar_processing/Clustering.hpp"
#include "sonar_processing/Denoising.hpp"
#include "sonar_processing/ImageUtils.hpp"
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_util/Converter.hpp"
#include "sonarlog_features/Application.hpp"

using namespace sonarlog_features;
using namespace sonar_processing;
using namespace sonar_processing::denoising;

int main(int argc, char const *argv[]) {

const std::string logfiles[] = {
    DATA_PATH_STRING + "/logs/gemini-harbor.3.log",
    // DATA_PATH_STRING + "/logs/gemini-marina.0.log",
    // "/arquivos/Logs/gemini/dataset_gustavo/logs/20160316-1127-06925_07750-gemini.0.log",
};

uint num_logfiles = sizeof(logfiles) / sizeof(std::string);
RLS rls(3);

for (size_t i = 0; i < num_logfiles; i++) {
    rock_util::LogReader reader(logfiles[i]);
    rock_util::LogStream stream = reader.stream("gemini.sonar_samples");

    base::samples::Sonar sample;
    while (stream.current_sample_index() < stream.total_samples()) {

        stream.next<base::samples::Sonar>(sample);

        /* cartesian properties */
        std::vector<float> bearings = rock_util::Utilities::get_radians(sample.bearings);
        float angle = bearings[bearings.size() - 1];
        uint32_t frame_height = 400;
        uint32_t frame_width = base::MathUtil::aspect_ratio_width(angle, frame_height);

        /* current frame */
        cv::Mat cart_raw = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);
        cart_raw.convertTo(cart_raw, CV_32F, 1.0 / 255);

        /* denoising */
        cv::Mat cart_denoised;
        cart_denoised = rls.sliding_window(cart_raw);

        /* roi masks */
        cv::Mat cart_roi_mask, polar_roi_mask, cart_image;
        preprocessing::extract_roi_masks(cart_denoised, bearings, sample.bin_count, sample.beam_count, cart_roi_mask, polar_roi_mask, 0.1);
        cart_denoised.copyTo(cart_image, cart_roi_mask);

        /* gradient */
        cart_image.convertTo(cart_image, CV_8U, 255);
        cv::Mat cart_aux, cart_grad;
        cv::boxFilter(cart_image, cart_aux, CV_8U, cv::Size(5, 5));
        preprocessing::gradient_filter(cart_aux, cart_grad);
        cv::normalize(cart_grad, cart_grad, 0, 255, cv::NORM_MINMAX);
        cv::boxFilter(cart_aux, cart_aux, CV_8U, cv::Size(30, 30));
        cart_grad -= cart_aux;
        cv::normalize(cart_grad, cart_grad, 0, 255, cv::NORM_MINMAX);
        cv::morphologyEx(cart_roi_mask, cart_roi_mask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)), cv::Point(-1, -1), 2);
        cv::Mat cart_grad2;
        cart_grad.copyTo(cart_grad2, cart_roi_mask);
        cv::medianBlur(cart_grad2, cart_grad2, 5);

        /* filtering */
        cv::Mat cart_filtered;
        cv::blur(cart_grad2, cart_filtered, cv::Size(5, 5));
        for (size_t i = 0; i < cart_filtered.rows; i++) {
            for (size_t j = 0; j < cart_filtered.cols; j++) {
                if (cart_filtered.at<uchar>(i,j) < 255 * 0.2)
                    cart_filtered.at<uchar>(i,j) = 0;
            }
        }

        /* threshold */
        cv::Mat cart_thresh;
        cv::threshold(cart_filtered, cart_thresh, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
        cv::morphologyEx(cart_roi_mask, cart_roi_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)), cv::Point(-1, -1), 2);
        preprocessing::remove_blobs(cart_thresh, cart_thresh, cv::Size(5, 5));

        cart_thresh.convertTo(cart_thresh, CV_32F, 1.0 / 255);
        cart_grad2.convertTo(cart_grad2, CV_32F, 1.0 / 255.0);

        /* output */
        cv::Mat out, out1, out2;
        out1.push_back(cart_raw);
        out1.push_back(cart_denoised);
        out2.push_back(cart_grad2);
        out2.push_back(cart_thresh);
        cv::hconcat(out1, out2, out);
        cv::resize(out, out, cv::Size(out.cols * 0.6, out.rows * 0.6));
        cv::imshow("out", out);
        cv::waitKey();
    }
    cv::waitKey(0);
}

return 0;
}
