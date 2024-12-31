
#include "LBDextractor.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

struct sort_lines_by_response
{
    inline bool operator()(const cv::line_descriptor::KeyLine &a, const cv::line_descriptor::KeyLine &b)
    {
        return ( a.response > b.response );
    }
};

LBDextractor::LBDextractor(int _nfeatures)
{
    nfeatures = _nfeatures;
}

void LBDextractor::operator()(const cv::Mat& image, const cv::Mat& mask, std::vector<cv::line_descriptor::KeyLine>& keylines, cv::Mat& descriptors)
{
    if(image.empty())
        return;
    assert( image.type () == CV_8UC1 );

    // detect line segment by LSD-detector
    DetectKeyLines(image, keylines);

    // Compute the LBD descriptors
    ComputeDescriptors(image, keylines, descriptors);
}

void LBDextractor::DetectKeyLines(const cv::Mat& image, std::vector<cv::line_descriptor::KeyLine>& keylines)
{
    cv::Ptr<cv::line_descriptor::LSDDetector> lsd = cv::line_descriptor::LSDDetector::createLSDDetector();

    lsd->detect(image, keylines, 1, 1);

    if (keylines.size() > nfeatures)
    {
        std::sort(keylines.begin(), keylines.end(), sort_lines_by_response());

        keylines.resize(nfeatures);

        for(int i=0; i<nfeatures; i++)
            keylines[i].class_id = i;
    }
}

void LBDextractor::ComputeDescriptors(const cv::Mat& image, std::vector<cv::line_descriptor::KeyLine>& keylines, cv::Mat& descriptors)
{
    descriptors = cv::Mat::zeros((int)keylines.size(), 32, CV_8UC1);

    // Create an LBD detector
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> lbd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();

    // Compute LBD
    lbd->compute(image, keylines, descriptors);
}