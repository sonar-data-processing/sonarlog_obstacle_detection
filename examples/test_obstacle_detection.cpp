#include <iostream>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "base/test_config.h"
#include "rock_util/LogReader.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonar_util/Converter.hpp"
#include "sonar_processing/ScanningHolder.hpp"

using namespace sonar_processing;

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
        ScanningHolder holder(600,600);

        while (stream.current_sample_index() < stream.total_samples()) {
            stream.next<base::samples::Sonar>(sample);
            holder.update(sample);

            cv::Mat cart_raw = holder.getCartImage();
            cv::Mat cart_mask = holder.getCartMask();

            std::cout << "=== IDX : " << stream.current_sample_index() << std::endl;
            std::cout << "=== BINS: " << sample.bin_count << std::endl;

            // output
            cv::imshow("cart_raw", cart_raw);
            cv::waitKey(5);
        }
    }
}
