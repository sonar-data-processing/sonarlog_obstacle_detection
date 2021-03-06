#include <iostream>
#include <deque>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "base/test_config.h"
#include "rock_util/LogReader.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonar_util/Converter.hpp"
#include "sonar_processing/ScanningHolder.hpp"

using namespace sonar_processing;

cv::Mat getSymmetricData(const cv::Mat& src) {
    cv::Mat left  = src(cv::Rect(0, 0, src.cols * 0.5, src.rows));
    cv::Mat right = src(cv::Rect(src.cols * 0.5, 0, src.cols * 0.5, src.rows));

    cv::Mat left_mirror;
    cv::flip(left, left_mirror, 1);

    cv::Mat out_right = 1 - (left_mirror + right);
    cv::medianBlur(out_right, out_right, 3);
    out_right.setTo(0, out_right < 0.8);

    cv::Mat out_left;
    cv::flip(out_right, out_left, 1);

    cv::Mat dst;
    cv::hconcat(out_left, out_right, dst);
    return dst;
}

void removeSparsenessBins(std::deque<base::samples::Sonar>& window, const base::samples::Sonar& sample) {
    if(window.empty() || (window.front().bin_count != sample.bin_count)) {
        window.clear();
    }

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
        DATA_PATH_STRING + "/logs/micron_fix.log",
    };

    uint num_logfiles = sizeof(logfiles) / sizeof(std::string);
    size_t start_index = (argc == 2) ? atoi(argv[1]) : 0;

    for (size_t i = 0; i < num_logfiles; i++) {
        rock_util::LogReader reader(logfiles[i]);
        rock_util::LogStream stream = reader.stream("micron_front.sonar_samples");
        stream.set_current_sample_index(start_index);

        base::samples::Sonar sample;
        ScanningHolder holder1(600, 600);
        ScanningHolder holder2(600, 600);
        std::deque<base::samples::Sonar> window;

        while (stream.current_sample_index() < stream.total_samples()) {
            stream.next<base::samples::Sonar>(sample);

            // update scanning holder
            holder1.update(sample);
            cv::Mat cart_raw = holder1.getCartImage();

            // remove sparseness bins
            removeSparsenessBins(window, sample);

            if (window.size() == 3) {
                base::samples::Sonar processed = window.back();
                window.pop_back();

                holder2.update(processed);
                cv::Mat cart_processed = holder2.getCartImage();
                cv::imshow("cart_processed", cart_processed);
            }



            // cv::Mat cart_mask = holder.getCartMask();

            // // remove symmetric data
            // cv::Mat cart_sym = getSymmetricData(cart_raw);
            // cv::Mat cart_out = cart_raw - cart_sym;
            // cart_out.setTo(0, cart_out < 0);

            // output
            cv::imshow("cart_raw", cart_raw);
            // cv::imshow("cart_sym", cart_sym);
            // cv::imshow("cart_out", cart_out);
            std::cout << "========== IDX   : " << stream.current_sample_index() << std::endl;
            std::cout << "========== BINS  : " << sample.bin_count << std::endl;
            std::cout << "========== RANGE : " << (sample.bin_count * 0.05) << "m" << std::endl;
            cv::waitKey(5);
        }
    }
}
