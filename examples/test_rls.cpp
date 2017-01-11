#include <iostream>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "base/test_config.h"
#include "rock_util/LogReader.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonar_processing/Denoising.hpp"
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_util/Converter.hpp"
#include "sonarlog_features/Application.hpp"

using namespace sonarlog_features;
using namespace sonar_processing;
using namespace sonar_processing::denoising;

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        "/arquivos/Logs/gemini/dataset_gustavo/logs/20160316-1127-06925_07750-gemini.0.log",
        // DATA_PATH_STRING + "/logs/gemini-marina.0.log",
        // DATA_PATH_STRING + "/logs/gemini-harbor.2.log",
    };

    uint32_t sz = sizeof(logfiles) / sizeof(std::string);
    RLS rls1, rls2(4), rls3;

    for (uint32_t i = 0; i < sz; i++) {
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

            /* current frame */
            cv::Mat input(sample.beam_count, sample.bin_count, CV_32F, (void*) sample.bins.data());

            /* denoising process */
            cv::Mat denoised_standard = rls1.infinite_window(input);
            cv::Mat denoised_fixed = rls2.sliding_window(input);
            cv::Mat denoised_adaptative = rls3.adaptative_window(input);

            /* output results */
            sample.bins.assign((float*) input.datastart, (float*) input.dataend);
            cv::Mat out1 = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);
            sample.bins.assign((float*) denoised_standard.datastart, (float*) denoised_standard.dataend);
            cv::Mat out2 = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);
            sample.bins.assign((float*) denoised_fixed.datastart, (float*) denoised_fixed.dataend);
            cv::Mat out3 = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);
            sample.bins.assign((float*) denoised_adaptative.datastart, (float*) denoised_adaptative.dataend);
            cv::Mat out4 = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);

            cv::Mat out;
            out1.push_back(out2);
            out3.push_back(out4);
            cv::hconcat(out1, out3, out);
            cv::resize(out, out, cv::Size(out.cols * 0.6, out.rows * 0.6));
            cv::imshow("out", out);
            cv::waitKey();
        }
    }

    return 0;
}
