#include <iostream>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "base/Plot.hpp"
#include "base/test_config.h"
#include "rock_util/LogReader.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonar_processing/Denoising.hpp"
#include "sonar_processing/ImageUtils.hpp"
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_util/Converter.hpp"
#include "sonarlog_features/Application.hpp"

using namespace sonarlog_features;
using namespace sonar_processing;
using namespace sonar_processing::denoising;

void find_peaks(cv::InputArray _src);

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        DATA_PATH_STRING + "/logs/gemini-marina.0.log",
        DATA_PATH_STRING + "/logs/gemini-harbor.2.log",
    };

    uint32_t sz = sizeof(logfiles) / sizeof(std::string);
    RLS rls(4);

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
            input.convertTo(input, CV_8U, 255);

            /* image enhancement */
            cv::Mat enhanced;
            preprocessing::adaptative_clahe(input, enhanced);

            // /* denoising process */
            // cv::Mat denoised;
            // rls.sliding_window(enhanced, denoised);
            //
            // /* roi frame */
            // cv::Mat roi = denoised(preprocessing::calc_horiz_roi(denoised));
            // cv::Mat diff = cv::Mat::zeros(cv::Size(denoised.cols - roi.cols, denoised.rows), CV_32F);
            // cv::hconcat(diff, roi, roi);
            //
            // /* segmentation */
            // find_peaks(roi);
            //
            // /* convert to cartesian plane */
            // input.convertTo(input, CV_32F, 1.0 / 255.0);
            // sample.bins.assign((float*) input.datastart, (float*) input.dataend);
            // cv::Mat out1 = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);
            // sample.bins.assign((float*) roi.datastart, (float*) roi.dataend);
            // cv::Mat out2 = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);
            //
            // cv::Mat out;
            // cv::hconcat(out1, out2, out);
            // cv::imshow("out", out);
            // cv::waitKey(30);
        }
    }

    return 0;
}

void find_peaks(cv::InputArray _src) {
    cv::Mat src = _src.getMat();
    if (src.depth() != CV_32F) src.convertTo(src, CV_32F, 1.0 / 255.0);

    std::vector<float> bins;
    for (size_t i = 0; i < src.rows; i++) {
        bins.assign((float*) src.row(i).datastart, (float*) src.row(i).dataend);
        float i_min = 2, pi_end = 0.4, pi_recursive = 0.6, h_search = 0.3;
        float i_max, h_total, h_end, h_current;

        for (size_t j = 0; j < bins.size(); j++) {
            // reinitialize i_min and i_max
            if (bins[j] < i_min) {
                i_min = bins[j];
                i_max = 0;
            }

            else {
                // update i_max
                if (bins[j] > i_max) {
                    i_max = bins[j];
                }

                else {
                    if (i_max > i_min) {
                        // update parameters
                        h_total = i_max - i_min;
                        h_end = h_total * pi_end;
                        h_current = bins[j] - i_min;

                        // finish peak analysis
                        if (h_current < h_end) {
                            // check if peak is accepted
                            if (h_total > h_search) {
                                float p_recursive = i_min + h_total * pi_recursive;
                                // extract segment
                            }
                            i_min = bins[j];
                            i_max = 0;
                        }
                    }
                }
            }
        }
    }
}
