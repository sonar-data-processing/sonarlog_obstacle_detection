#include <iostream>
#include <deque>
#include <base/samples/Sonar.hpp>
#include "base/test_config.h"
#include "rock_util/LogReader.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonar_util/Converter.hpp"
#include "sonar_processing/ScanningHolder.hpp"

using namespace sonar_processing;

void removeSparsenessBins(std::deque<base::samples::Sonar>& window, const base::samples::Sonar& sample) {
    if(window.empty() || (window.front().bin_count != sample.bin_count))
        window.clear();

    window.push_front(sample);

    if (window.size() == 3) {
        base::samples::Sonar last    = window.at(0);
        base::samples::Sonar current = window.at(1);
        base::samples::Sonar next    = window.at(2);

        for (size_t i = 0; i < current.bin_count; i++) {
            if(current.bins[i] && !last.bins[i] && !next.bins[i]) {
                current.bins[i] = 0;
            }
        }

        window.erase(window.begin() + 1);
        window.insert(window.begin() + 1, current);
    }
}


int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        DATA_PATH_STRING + "/logs/ssiv/ssiv_20170511.0.log",
    };

    uint num_logfiles = sizeof(logfiles) / sizeof(std::string);
    size_t start_index = (argc == 2) ? atoi(argv[1]) : 0;

    for (size_t i = 0; i < num_logfiles; i++) {
        rock_util::LogReader reader(logfiles[i]);
        rock_util::LogStream stream = reader.stream("micron_front.sonar_samples");
        stream.set_current_sample_index(start_index);

        base::samples::Sonar sample;
        base::Angle left_limit  = base::Angle::fromDeg(-45.0);
        base::Angle right_limit = base::Angle::fromDeg( 45.0);
        ScanningHolder holder1(600, 600, left_limit, right_limit);
        ScanningHolder holder2(600, 600, left_limit, right_limit);
        std::deque<base::samples::Sonar> window;

        while (stream.current_sample_index() < stream.total_samples()) {
            stream.next<base::samples::Sonar>(sample);

            // update scanning holder
            holder1.update(sample);
            cv::Mat cart_raw = holder1.getCartImage();
            cart_raw = cart_raw(cv::Rect(0, 0, cart_raw.cols, cart_raw.rows * 0.5));

            // remove sparseness bins
            removeSparsenessBins(window, sample);
            if (window.size() == 3) {
                base::samples::Sonar processed = window.back();
                window.pop_back();

                holder2.update(processed);
                cv::Mat cart_processed = holder2.getCartImage();
                cart_processed = cart_processed(cv::Rect(0, 0, cart_processed.cols, cart_processed.rows * 0.5));
                cv::imshow("cart_processed", cart_processed);
            }

            // output
            cv::imshow("cart_raw", cart_raw);
            std::cout   << "===== [IDX, BINS, RANGE] : ["
                        << stream.current_sample_index() << ", "
                        << sample.bin_count << ", "
                        << sample.getBinStartDistance(sample.bin_count) << "m]" << std::endl;
            cv::waitKey();
        }
    }
}
