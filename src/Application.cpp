#include <opencv2/opencv.hpp>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "sonar_util/Converter.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonarlog_obstacle_detection/Application.hpp"
#include "sonar_processing/ImageUtil.hpp"
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_processing/QualityMetrics.hpp"

using namespace sonar_processing;

namespace sonarlog_obstacle_detection {

Application *Application::instance_ = NULL;

Application*  Application::instance() {
    if (!instance_){
        instance_ = new Application();
    }
    return instance_;
}

void Application::init(const std::string& filename, const std::string& stream_name) {
    reader_.reset(new rock_util::LogReader(filename));
    plot_.reset(new base::Plot());
    stream_ = reader_->stream(stream_name);
}

void Application::process_next_sample() {
    base::samples::Sonar sample;
    stream_.next<base::samples::Sonar>(sample);
}

void Application::process_logfile() {
    rls.setWindow_size(4);
    stream_.reset();
    while (stream_.current_sample_index() < stream_.total_samples()) process_next_sample();
    cv::waitKey();
}

void Application::plot(cv::Mat mat) {
    (*plot_)(image_util::mat2vector<float>(mat));
}

}
