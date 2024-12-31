#ifndef LBDMATCHER_H
#define LBDMATCHER_H

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

#include "KeyFrame.h"

class LBDmatcher
{ 
public:
    LBDmatcher(float nnratio=0.6, bool checkOri=true);

    // Matching for the Map Initialization
    int SearchForInitialization(KeyFrame &F1, KeyFrame &F2,
                                std::vector<cv::line_descriptor::KeyLine*> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize);
    
    // Computes the Hamming distance between two LBD descriptors 
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

public:
    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;

protected:
    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

    float mfNNratio;
    bool mbCheckOrientation;
};

#endif // LBDMATCHER_H
