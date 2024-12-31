#ifndef VIEWER_H
#define VIEWER_H

#include "MapDrawer.h"

class Viewer
{ 
public:
    Viewer(MapDrawer* pMapDrawer, const string &strSettingPath, KeyFrame* showFrame = NULL);

    void Run();

    void RequestOptimize();
    bool isRequestOptimize(); 

    void SetOptimizeFinish();
    void RequestFinish();
    bool isFinished();

private:
    MapDrawer* mpMapDrawer;

    KeyFrame* mpFrame;

    float mViewpointX;
    float mViewpointY; 
    float mViewpointZ; 
    float mViewpointF;

    std::mutex mMutexOptimize;
    bool mbOptimizeRequested;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;
};

#endif // VIEWER_H
