#ifndef LBDEXTRACTOR_H
#define LBDEXTRACTOR_H

#include <vector>
#include <list>

#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

class LBDextractor
{
public:
    LBDextractor(int _nfeatures);

    ~LBDextractor(){}

    void operator() (const cv::Mat& image, const cv::Mat& mask, std::vector<cv::line_descriptor::KeyLine>& keylines, cv::Mat& descriptors);

protected:
    cv::Mat mImage;

    void DetectKeyLines(const cv::Mat& image, std::vector<cv::line_descriptor::KeyLine>& keylines);

    void ComputeDescriptors(const cv::Mat& image, std::vector<cv::line_descriptor::KeyLine>& keyLines, cv::Mat& descriptors);
    
    int nfeatures;
};

#endif // LBDEXTRACTOR_H
