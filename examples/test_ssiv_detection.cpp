#include <iostream>
#include <deque>
#include <base/samples/Sonar.hpp>
#include "base/test_config.h"
#include "rock_util/LogReader.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonar_util/Converter.hpp"
#include "sonar_processing/ScanningHolder.hpp"
#include <opencv2/opencv.hpp>

using namespace sonar_processing;

void removeSparsenessBins(std::deque<base::samples::Sonar>& window, const base::samples::Sonar& sample) {
    if(window.empty() || (window.front().bin_count != sample.bin_count))
        window.clear();

    window.push_front(sample);

    if (window.size() == 3) {
        base::samples::Sonar last    = window.at(0);
        base::samples::Sonar current = window.at(1);
        base::samples::Sonar next    = window.at(2);

        for (size_t i = 1; i < current.bin_count - 1; i++) {
            if(current.bins[i] && ((!last.bins[i] && !next.bins[i]) || (!current.bins[i - 1] && !current.bins[i+1] ))) {
                current.bins[i] = 0;
            }
        }

        window.erase(window.begin() + 1);
        window.insert(window.begin() + 1, current);
    }
}

cv::Mat removeSymmetricData(const cv::Mat& src) {
    cv::Mat left  = src(cv::Rect(0, 0, src.cols * 0.5, src.rows));
    cv::Mat right = src(cv::Rect(src.cols * 0.5, 0, src.cols * 0.5, src.rows));

    cv::Mat left_mirror;
    cv::flip(left, left_mirror, 1);

    cv::Mat out_right = 1 - (left_mirror + right);
    cv::medianBlur(out_right, out_right, 3);
    out_right.setTo(0, out_right < 0.8);

    cv::Mat out_left;
    cv::flip(out_right, out_left, 1);

    cv::Mat sym;
    cv::hconcat(out_left, out_right, sym);

    cv::Mat dst = src - sym;
    dst.setTo(0, dst < 0);

    return dst;
}

void extractRoi(const cv::Mat& src, cv::Mat& dst, float min_range, float max_range, const base::samples::Sonar& sonar) {
    float total_range = sonar.getBinStartDistance(sonar.bin_count);
    if(max_range > total_range) max_range = total_range;

    float min_bin = sonar.bin_count * min_range / total_range;
    float max_bin = sonar.bin_count * max_range / total_range;

    float resolution = src.rows / (float) sonar.bin_count;
    int row0 = src.rows - resolution * min_bin;
    int row1 = src.rows - resolution * max_bin;

    dst = src.clone();
    dst.rowRange(0, row1).setTo(0);
    dst.rowRange(row0, dst.rows).setTo(0);
}

cv::Rect getMaskLimits(const cv::Mat& mask) {
    // check if mask is valid
    size_t mask_pixels = cv::countNonZero(mask);
    if(!mask_pixels) return cv::Rect();

    // find mask limits
    cv::Mat points;
    cv::findNonZero(mask, points);
    return cv::boundingRect(points);
}

double euclideanDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    cv::Point2f diff = p1 - p2;
    return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}

base::Vector2d getWorldPoint(const cv::Point2f& p, cv::Size size, float range) {
    // convert from image to cartesian coordinates
    cv::Point2f origin(size.width / 2, size.height - 1);
    cv::Point2f q(origin.y - p.y, origin.x - p.x);

    // sonar resolution
    float sonar_resolution = range / size.height;

    // 2d world coordinates
    float x = q.x * sonar_resolution;
    base::Angle angle = base::Angle::fromRad(atan2(q.y, q.x));
    float y = tan(angle.rad) * x;

    // output
    return base::Vector2d(x, y);
}

void getTargetDistance(const cv::Mat& src) {
    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, CV_BGR2GRAY);

    cv::Rect bbox = getMaskLimits(src_gray);
    double closest_distance = 100000;
    cv::Point closest;

    cv::Point2f origin(src.cols / 2, src.rows - 1);
    for (size_t i = 0; i < 3; i++) {
        cv::Point2f p(bbox.x + i * bbox.width / 2, bbox.y + bbox.height);
        double distance = euclideanDistance(p, origin);
        if(distance < closest_distance) {
            closest_distance = distance;
            closest = p;
        }
    }

    cv::Mat tst = cv::Mat::zeros(src.size(), src.type());
    cv::rectangle(tst, bbox, cv::Scalar(255,0,0));
    cv::circle(tst, closest, 1, cv::Scalar(0,255,255));
    base::Vector2d ssiv = getWorldPoint(closest, src.size(), 20);
    std::cout << "SSIV: " << ssiv << std::endl;
    cv::imshow("tst", tst);
}

cv::Mat findBiggestBlob(const cv::Mat& src, int minPxContour) {
    cv::Mat src_8u;
    src.convertTo(src_8u, CV_8U, 255);

    std::vector<std::vector<cv::Point> > contours, biggest_contour;
    cv::findContours(src_8u.clone(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    for (size_t j = 0; j < contours.size(); j++) {
        size_t pxContour = contours[j].size();
        if (pxContour > minPxContour) {
            if(!biggest_contour.empty() && (pxContour > biggest_contour[0].size())) {
                biggest_contour[0] = contours[j];
            } else {
                biggest_contour.push_back(contours[j]);
            }
        }
    }

    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    cv::drawContours(dst, biggest_contour, -1, cv::Scalar(0,0,255), 1);

    if(!biggest_contour.empty()) {
        cv::Rect bounding_rect = cv::boundingRect(biggest_contour[0]);
        cv::rectangle(dst, bounding_rect, cv::Scalar(0,255,0));
        getTargetDistance(dst);
    }
    return dst;
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
        ScanningHolder holder1(800, 800, left_limit, right_limit);
        ScanningHolder holder2(800, 800, left_limit, right_limit);

        std::deque<base::samples::Sonar> window;
        int min_range = 1, max_range = 7;

        while (stream.current_sample_index() < stream.total_samples()) {
            stream.next<base::samples::Sonar>(sample);
            holder1.update(sample);

            // raw data
            cv::Mat cart_raw = holder1.getCartImage();
            cart_raw = cart_raw(cv::Rect(0, 0, cart_raw.cols, cart_raw.rows * 0.5));

            // mask
            cv::Mat cart_mask = holder1.getCartMask();
            cart_mask = cart_mask(cv::Rect(0, 0, cart_mask.cols, cart_mask.rows * 0.5));

            // remove sparseness bins
            removeSparsenessBins(window, sample);
            if(window.size() != 3) continue;

            // update scanning holder
            holder2.update(window.back());
            window.pop_back();
            cv::Mat cart_fltr = holder2.getCartImage();
            cart_fltr = cart_fltr(cv::Rect(0, 0, cart_fltr.cols, cart_fltr.rows * 0.5));
            cv::blur(cart_fltr, cart_fltr, cv::Size(3, 3));
            cart_fltr = removeSymmetricData(cart_fltr);

            // extract roi
            cv::Mat cart_roi;
            extractRoi(cart_fltr, cart_roi, min_range, max_range, sample);

            // binary image
            cv::Mat cart_thresh;
            cv::threshold(cart_roi, cart_thresh, 0.1, 1.0, CV_THRESH_BINARY);
            cv::morphologyEx(cart_thresh, cart_thresh, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1, -1), 2);

            // biggest contour
            cv::Mat dst = findBiggestBlob(cart_thresh, 100);

            // output
            cv::imshow("cart_raw", cart_raw);
            // cv::imshow("cart_mask", cart_mask);
            cv::imshow("cart_fltr", cart_fltr);
            cv::imshow("cart_roi", cart_roi);
            // cv::imshow("cart_thresh", cart_thresh);
            cv::imshow("dst", dst);

            // for (size_t i = 0; i < contours.size(); i++) {
            //     std::cout << "Contours[" << i << "] = " << contours[i].size() << std::endl;
            // }

            std::cout << "========== IDX   : " << stream.current_sample_index() << std::endl;
            // std::cout << "Bins: " << cv::Mat(sample.bins).t() << std::endl;
            cv::waitKey(5);
        }
    }
}
