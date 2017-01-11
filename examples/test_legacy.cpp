#include <iostream>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "base/test_config.h"
#include "rock_util/LogReader.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonar_processing/ImageUtils.hpp"
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_util/Converter.hpp"
#include "sonarlog_features/Application.hpp"

using namespace sonarlog_features;
using namespace sonar_processing;

cv::Mat create_mask_new(cv::Mat grad, cv::Mat roi);

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        DATA_PATH_STRING + "/logs/gemini-jequitaia.0.log",
        DATA_PATH_STRING + "/logs/gemini-jequitaia.4.log",
        DATA_PATH_STRING + "/logs/gemini-ferry.0.log",
        DATA_PATH_STRING + "/logs/gemini-ferry.3.log",
    };

    uint32_t log_count = sizeof(logfiles) / sizeof(std::string);

    for (uint32_t i = 0; i < log_count; i++) {
        rock_util::LogReader reader(logfiles[i]);
        rock_util::LogStream stream = reader.stream("gemini.sonar_samples");

        base::samples::Sonar sample;
        while (stream.current_sample_index() < stream.total_samples()) {
            stream.next<base::samples::Sonar>(sample);

            /* cartesian properties */
            std::vector<float> bearings = rock_util::Utilities::get_radians(sample.bearings);
            float angle = bearings[bearings.size()-1];
            uint32_t frame_height = 400;
            uint32_t frame_width = base::MathUtil::aspect_ratio_width(angle, frame_height);

            /* roi */
            cv::Mat src(sample.beam_count, sample.bin_count, CV_32F, (void*) sample.bins.data());
            cv::Mat roi = src(preprocessing::calc_horiz_roi(src));
            roi.convertTo(roi, CV_8U, 255);

            /* image enhancement */
            cv::Mat enhanced;
            preprocessing::adaptative_clahe(roi, enhanced);

            /* denoising */
            cv::Mat filtered;
            cv::GaussianBlur(enhanced, filtered, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);
            if (filtered.depth() != CV_8U) filtered.convertTo(filtered, CV_8U, 255);

            /* gradient */
            cv::Mat grad;
            preprocessing::gradient_filter(filtered, grad);
            cv::normalize(grad, grad, 0, 255, cv::NORM_MINMAX);

            /* mask */
            cv::Mat mask = create_mask_new(grad, filtered);
            cv::Mat segmented;
            enhanced(cv::Rect(0, 0, mask.cols, mask.rows)).copyTo(segmented, mask);

            cv::Mat diff_cols_ini = cv::Mat::zeros(cv::Size(src.cols - roi.cols, src.rows), CV_8U);
            cv::Mat diff_cols_end = cv::Mat::zeros(cv::Size(roi.cols - mask.cols, roi.rows), CV_8U);

            cv::Mat dst;
            cv::hconcat(diff_cols_ini, segmented, dst);
            cv::hconcat(dst, diff_cols_end, dst);

            /* convert back to cartesian plane */
            dst.convertTo(dst, CV_32F, 1.0 / 255.0);
            sample.bins.assign((float*) dst.datastart, (float*) dst.dataend);

            /* show results */
            cv::Mat output = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);
            cv::imshow("output", output);
            cv::waitKey(30);
        }
        cv::waitKey();
    }

    return 0;
}

cv::Mat create_mask_new(cv::Mat grad, cv::Mat roi) {
    cv::Mat thresh;
    cv::threshold(grad, thresh, image_utils::otsu_thresh_8u(grad) * 0.6, 255, CV_THRESH_BINARY);
    preprocessing::remove_blobs(thresh, thresh, cv::Size(20, 20));

    /* segmentation - mask */
    cv::Mat src_low_cols_removed;
    preprocessing::remove_low_intensities_columns(roi, src_low_cols_removed);

    cv::Mat mask, shadow_mask;
    preprocessing::weak_target_thresholding(src_low_cols_removed, mask);
    return mask;
}
