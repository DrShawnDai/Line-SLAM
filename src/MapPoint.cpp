#include "MapPoint.h"

MapPoint::MapPoint(const cv::Mat &Pos, size_t id)
{
    Pos.copyTo(mWorldPos);
}

MapPoint::MapPoint(KeyFrame* kf1, KeyFrame* kf2, size_t kpID1, size_t kpID2)
{
    cv::Mat kp_img1 = (cv::Mat_<float>(2, 1) << kf1->GetKeyPoint(kpID1)->x, kf1->GetKeyPoint(kpID1)->y);
    cv::Mat kp_img2 = (cv::Mat_<float>(2, 1) << kf2->GetKeyPoint(kpID2)->x, kf2->GetKeyPoint(kpID2)->y);
    
    // 将像素坐标转换至相机坐标
    cv::Mat kp_cam1 = kf1->pixelToCam(kp_img1);
    cv::Mat kp_cam2 = kf2->pixelToCam(kp_img2);
    
    // 三角化重建
    Triangulate(kp_cam1, kp_cam2, kf1->GetPose(), kf2->GetPose(), mWorldPos);
}

cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexMapPoint);
    return mWorldPos.clone();
}

void MapPoint::UpdateWorldPos(const Vector3d pos)
{
    unique_lock<mutex> lock(mMutexMapPoint);

    mWorldPos.at<float>(0) = float(pos(0));
    mWorldPos.at<float>(1) = float(pos(1));
    mWorldPos.at<float>(2) = float(pos(2));
}

cv::Mat MapPoint::ProjectToCamera(KeyFrame* kf)
{
    cv::Mat Rcw = kf->GetRotation();
    cv::Mat tcw = kf->GetTranslation();

    cv::Mat pixel = kf->K * (Rcw * this->GetWorldPos() + tcw);
    pixel = pixel / pixel.at<float>(2);
    return pixel.clone();
}

void MapPoint::Triangulate(const cv::Mat &kp1, const cv::Mat &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4, 4, CV_32F);

    A.row(0) = kp1.at<float>(0) * P1.row(2) - P1.row(0);
    A.row(1) = kp1.at<float>(1) * P1.row(2) - P1.row(1);
    A.row(2) = kp2.at<float>(0) * P2.row(2) - P2.row(0);
    A.row(3) = kp2.at<float>(1) * P2.row(2) - P2.row(1);

    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    x3D = vt.row(3).t(); // Multiple View Geometry in Computer Vision (Page 593)
    x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

    return;
}
