#include "Viewer.h"

Viewer::Viewer(MapDrawer* pMapDrawer, const string &strSettingPath, KeyFrame* showFrame):
    mpMapDrawer(pMapDrawer), mpFrame(showFrame), 
    mbFinishRequested(false), mbFinished(false), mbOptimizeRequested(false)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    mViewpointX = fSettings["Viewer.ViewpointX"];
    mViewpointY = fSettings["Viewer.ViewpointY"];
    mViewpointZ = fSettings["Viewer.ViewpointZ"];
    mViewpointF = fSettings["Viewer.ViewpointF"];
}

void Viewer::Run()
{
    mbFinished = false;

    pangolin::CreateWindowAndBind("Line-BA: Map Viewer", 1024, 768);

    // 3D Mouse handler requires depth testing to be enabled 
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175)); 
    pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true); 
    pangolin::Var<bool> menuShowLines("menu.Show Lines", true, true); 
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true); 
    pangolin::Var<bool> menuRequestOptimize("menu.Optimization", false, false); 
    pangolin::Var<bool> menuRequestFinish("menu.Finish", false, false);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000), 
                pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc;
    mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        mpMapDrawer->DrawCurrentCamera(Twc);

        if(menuShowKeyFrames)
            mpMapDrawer->DrawKeyFrames(menuShowKeyFrames);

        if(menuShowPoints)
            mpMapDrawer->DrawMapPoints();

        if(menuShowLines)
            mpMapDrawer->DrawMapLines();

        if(menuRequestOptimize)
        {
            RequestOptimize();
            menuRequestOptimize = false;
        }

        pangolin::FinishFrame();

        if(mpFrame != NULL)
        {
            cv::imshow("re-projection", mpFrame->GetImage());
            cv::waitKey(50);
        }

        if(menuRequestFinish)
            RequestFinish();

        if(CheckFinish())
            break;
    }

    SetFinish();
}

void Viewer::RequestOptimize()
{
    unique_lock<mutex> lock(mMutexOptimize);
    mbOptimizeRequested = true;
}

bool Viewer::isRequestOptimize()
{
    unique_lock<mutex> lock(mMutexOptimize);
    return mbOptimizeRequested;
}

void Viewer::SetOptimizeFinish()
{
    unique_lock<mutex> lock(mMutexOptimize);
    mbOptimizeRequested = false;
}

void Viewer::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}
