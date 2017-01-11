#include <iostream>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "base/test_config.h"
#include "rock_util/LogReader.hpp"
#include "rock_util/Utilities.hpp"
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_util/Converter.hpp"
#include "sonarlog_features/Application.hpp"

using namespace sonarlog_features;
using namespace sonar_processing;

static base::Plot plot;

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
            cv::Mat cart_image = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);

            /* roi masks - cartesian and polar planes */
            cv::Mat cart_roi_mask, polar_roi_mask, cart_roi;
            preprocessing::extract_roi_masks(cart_image, bearings, sample.bin_count, sample.beam_count, cart_roi_mask, polar_roi_mask);
            cart_image.copyTo(cart_roi, cart_roi_mask);

            /* output */
            cv::Mat out;
            cv::hconcat(cart_image, cart_roi, out);
            cv::imshow("out", out);
            cv::waitKey(30);
        }
        cv::waitKey();
    }

    return 0;
}
