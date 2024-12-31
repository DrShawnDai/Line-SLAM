#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include "Map.h"
#include <pangolin/pangolin.h>

class MapDrawer
{
public:
    MapDrawer(Map* pMap, const string &strSettingPath);

    void DrawMapPoints(); 
    void DrawMapLines();
    void DrawKeyFrames(const bool bDrawKF);

    void SetCurrentCameraPose(const cv::Mat &Tcw);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc) ;

    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

private:
    Map* mpMap;

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mCameraSize;
    float mCameraLineWidth;
    float mPointSize;

    cv::Mat mCameraPose;
    std::mutex mMutexCamera;
};

#endif // MAPDRAWER_H
