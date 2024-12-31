#include "MapLine.h"

MapLine::MapLine(const cv::Mat &Plucker, const cv::Mat &Points)
{
    Plucker.copyTo(mPluckerCoord);
    
    Points.copyTo(mEndPoints);
}

MapLine::MapLine(KeyFrame* kf1, KeyFrame* kf2, size_t lineID1, size_t lineID2, ConstuctType type) :
    mpFirstObsKF(kf1), mIndexinFirstKF(lineID1)
{
    cv::Mat sta_img1 = (cv::Mat_<float>(2, 1) << kf1->GetKeyLine(lineID1)->startPointX, kf1->GetKeyLine(lineID1)->startPointY);
    cv::Mat end_img1 = (cv::Mat_<float>(2, 1) << kf1->GetKeyLine(lineID1)->endPointX,   kf1->GetKeyLine(lineID1)->endPointY);

    cv::Mat sta_img2 = (cv::Mat_<float>(2, 1) << kf2->GetKeyLine(lineID2)->startPointX, kf2->GetKeyLine(lineID2)->startPointY);
    cv::Mat end_img2 = (cv::Mat_<float>(2, 1) << kf2->GetKeyLine(lineID2)->endPointX,   kf2->GetKeyLine(lineID2)->endPointY);

    // 将像素坐标转换至相机归一化坐标
    cv::Mat P_sta_c1 = kf1->pixelToCam(sta_img1);
    cv::Mat P_end_c1 = kf1->pixelToCam(end_img1);
    cv::Mat P_sta_c2 = kf2->pixelToCam(sta_img2);
    cv::Mat P_end_c2 = kf2->pixelToCam(end_img2);

    if(type == BY_TWOPLANES) //通过平面相交确定空间直线
    {
        // 1. 将归一化坐标转至世界坐标 Pw = Rwc * (Pc - tcw)
        cv::Mat P_sta_c1_w = kf1->camToWorld(P_sta_c1);
        cv::Mat P_end_c1_w = kf1->camToWorld(P_end_c1);

        cv::Mat P_sta_c2_w = kf2->camToWorld(P_sta_c2);
        cv::Mat P_end_c2_w = kf2->camToWorld(P_end_c2);

        cv::Mat zeros = cv::Mat::zeros(3,1,CV_32F);
        cv::Mat twc1 = kf1->camToWorld(zeros);
        cv::Mat twc2 = kf2->camToWorld(zeros);

        // 2. 构建两平面
        cv::Mat plane1 = reconstructPlane(P_sta_c1_w, P_end_c1_w, twc1);
        cv::Mat plane2 = reconstructPlane(P_sta_c2_w, P_end_c2_w, twc2);

        // 3. 求两平面交线（普吕克矩阵）
        cv::Mat plucker_mat = PluckerFromTwoPlanes(plane1, plane2);

        // 4. 从平面交线普吕克矩阵转化为普吕克坐标(n， d)
        cv::Mat plucker_coord = PluckerMatToCoordinate(plucker_mat);
        plucker_coord.rowRange(0, 3).copyTo(mPluckerCoord.rowRange(0, 3));
        plucker_coord.rowRange(3, 6).copyTo(mPluckerCoord.rowRange(3, 6));

        // 端点初始化（用 kf1 观测到的端点初始化，后续不断迭代优化）
        cv::Mat P_sta_w = PluckerFromTwoPoints(twc1, P_sta_c1_w) * plane2;
        cv::Mat P_end_w = PluckerFromTwoPoints(twc1, P_end_c1_w) * plane2;
        P_sta_w = P_sta_w.rowRange(0, 3) / P_sta_w.at<float>(3);
        P_end_w = P_end_w.rowRange(0, 3) / P_end_w.at<float>(3);

        P_sta_w.copyTo(mEndPoints.rowRange(0, 3).col(0));
        P_end_w.copyTo(mEndPoints.rowRange(0, 3).col(1));

        // 普吕克坐标转化为正交表示
        ComputeOrthByPlucker();
    }
    else if(type == BY_ENDPOINTS) // 通过两个端点三角化重建确定空间直线（！要求端点正确匹配！）
    {
        cv::Mat x3D1;
        cv::Mat x3D2;
        Triangulate(P_sta_c1, P_sta_c2, kf1->GetPose(), kf2->GetPose(), x3D1);
        Triangulate(P_end_c1, P_end_c2, kf1->GetPose(), kf2->GetPose(), x3D2);

        // 通过两个端点空间直线的普吕克矩阵
        cv::Mat plucker_mat = PluckerFromTwoPoints(x3D1, x3D2);

        // 普吕克矩阵转化为普吕克坐标(n，d)
        // 注意：由点构建的直线普吕克坐标，与由平面构建的普吕克坐标，n、 d 位置互换
        cv::Mat plucker_coord = PluckerMatToCoordinate(plucker_mat);
        plucker_coord.rowRange(0, 3).copyTo(mPluckerCoord.rowRange(3, 6)); 
        plucker_coord.rowRange(3, 6).copyTo(mPluckerCoord.rowRange(0, 3));

        // 端点初始化
        x3D1.copyTo(mEndPoints.rowRange(0, 3).col(0));
        x3D2.copyTo(mEndPoints.rowRange(0, 3).col(1));

        // 普吕克坐标转化为正交表示
        ComputeOrthByPlucker();
    }
}

cv::Mat MapLine::GetNormal()
{
    unique_lock<mutex> lock(mMutexMapLine);
    return mPluckerCoord.rowRange(0, 3).clone();
}

cv::Mat MapLine::GetDirect()
{
    unique_lock<mutex> lock(mMutexMapLine);
    return mPluckerCoord.rowRange(3, 6).clone();
}

cv::Mat MapLine::GetOrthonormal()
{
    unique_lock<mutex> lock(mMutexMapLine);
    return mOrthonormal.clone();
}

cv::Mat MapLine::GetEndPoints()
{
    unique_lock<mutex> lock(mMutexMapLine);
    return mEndPoints.clone();
}

void MapLine::UpdateEndPoints()
{
    // 更新后的直线普吕克矩阵
    cv::Mat Lw = PluckerCoordinateToMat(mPluckerCoord);

    // 第一次观测到这条线特征
    cv::line_descriptor::KeyLine *kl = mpFirstObsKF->GetKeyLine(mIndexinFirstKF);
    cv::Mat sta_img = (cv::Mat_<float>(2, 1) << kl->startPointX, kl->startPointY);
    cv::Mat end_img = (cv::Mat_<float>(2, 1) << kl->endPointX,   kl->endPointY);

    // 将像素坐标转换至相机归一化坐标
    cv::Mat sta_cam = mpFirstObsKF->pixelToCam(sta_img);
    cv::Mat end_cam = mpFirstObsKF->pixelToCam(end_img);

    // 计算直线的法向
    cv::Mat ln = (sta_cam.cross(end_cam)).rowRange(0, 2);
    float norm = sqrt(pow(ln.at<float>(0), 2) + pow(ln.at<float>(1), 2));
    ln = ln / norm;

    // 直线垂直方向上移动一个单位
    cv::Mat sta_c_delt = (cv::Mat_<float>(3, 1) <<  sta_cam.at<float>(0) + ln.at<float>(0), 
                                                    sta_cam.at<float>(1) + ln.at<float>(1),
                                                    1.0f);
    cv::Mat end_c_delt = (cv::Mat_<float>(3, 1) <<  end_cam.at<float>(0) + ln.at<float>(0),
                                                    end_cam.at<float>(1) + ln.at<float>(1),
                                                    1.0f);
    cv::Mat cam = cv::Mat::zeros(3, 1, CV_32F);

    // 构建两平面
    cv::Mat plane1_c = reconstructPlane(cam, sta_cam, sta_c_delt);
    cv::Mat plane2_c = reconstructPlane(cam, end_cam, end_c_delt);

    cv::Mat plane1_w = mpFirstObsKF->GetPose().t() * plane1_c;
    cv::Mat plane2_w = mpFirstObsKF->GetPose().t() * plane2_c;

    // 优化后的直线与两平面相交得到更新后的端点
    cv::Mat e1 = Lw * plane1_w;
    cv::Mat e2 = Lw * plane2_w;

    e1 = e1.rowRange(0,3)/e1.at<float>(3);
    e2 = e2.rowRange(0,3)/e2.at<float>(3);

    e1.copyTo(mEndPoints.rowRange(0, 3).col(0));
    e2.copyTo(mEndPoints.rowRange(0, 3).col(1));
}

void MapLine::UpdateByOrth(const Vector4d orth)
{
    unique_lock<mutex> lock(mMutexMapLine);

    cv::eigen2cv(orth, mOrthonormal);

    ComputePluckerByOrth();

    UpdateEndPoints() ;
}

void MapLine::Triangulate(const cv::Mat &kp1, const cv::Mat &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4, 4, CV_32F);

    A.row(0) = kp1.at<float>(0) * P1.row(2) - P1.row(0);
    A.row(1) = kp1.at<float>(1) * P1.row(2) - P1.row(1);
    A.row(2) = kp2.at<float>(0) * P2.row(2) - P2.row(0);
    A.row(3) = kp2.at<float>(1) * P2.row(2) - P2.row(1);

    // Multiple View Geometry in Computer Vision (Page 593)
    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV) ;
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3) / x3D.at<float>(3);

    return;
}

cv::Mat MapLine::reconstructPlane(const cv::Mat &P1, const cv::Mat &P2, const cv::Mat &P3)
{
    if(0)
    {   // 方法（一） SVD 求解
        cv::Mat A = cv::Mat::ones (3, 4, CV_32F);
        A.row(0).colRange(0, 3) = P1.t();
        A.row(1).colRange(0, 3) = P2.t();
        A.row(2).colRange(0, 3) = P3.t();

        cv::Mat u, w, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        cv::Mat plane = vt.row(3).t();
        plane = plane / plane.at<float>(3);
        return plane.clone();
    }
    else if(0)
    {   // 方法（二） Ax = b, x = A^(-1) * b
        cv::Mat A = cv::Mat::zeros(3, 3, CV_32F);
        A.row(0) = P1.t();
        A.row(1) = P2.t();
        A.row(2) = P3.t();

        cv::Mat A_inv;
        cv::invert(A, A_inv);

        cv::Mat ones = cv::Mat::ones(3, 1, CV_32F);
        cv::Mat plane = cv::Mat::ones(4, 1, CV_32F);
        plane.rowRange(0, 3) = -A_inv * ones;
        return plane.clone();
    }
    else
    {   // 方法（三）
        // 三点确定一个平面 a(x-x0) + b(y-y0) + c(z-z0) = 0
        // --> ax + by + cr + d = 0, d = -(a x0 + b y0 + c z0)
        // 平面通过点 (x0, y0, z0）以及垂直于平面的法线（a,b,c）来得到
        // (a,b,c)^T = vector(AO) × vector(BO), d = O.dot(cross(AO, BO))
        cv::Mat plane = cv::Mat::zeros(4, 1, CV_32F);

        cv::Mat normal = (P1 - P3).cross(P2 - P3);
        normal.copyTo(plane.rowRange(0, 3));

        // d = - x3.dot( (x1-×3).cross(x2-x3) ) = - x3.dot( x1.cross(x2) ) 
        plane.at<float>(3) = - P3.dot(P1.cross(P2));
        return plane;
    }
}

cv::Mat MapLine::PluckerFromTwoPlanes(const cv::Mat &plane1, const cv::Mat &plane2)
{
    cv::Mat plucker_mat = plane1 * plane2.t() - plane2 * plane1.t();
    return plucker_mat.clone();
}

cv::Mat MapLine::PluckerFromTwoPoints(const cv::Mat &x3D1, const cv::Mat &x3D2)
{
    cv::Mat p3D1 = cv::Mat::ones(4, 1, CV_32F);
    cv::Mat p3D2 = cv::Mat::ones(4, 1, CV_32F);
    x3D1.copyTo(p3D1.rowRange(0, 3));
    x3D2.copyTo(p3D2.rowRange(0, 3));
    cv::Mat plucker_mat = p3D1 * p3D2.t() - p3D2 * p3D1.t();

    return plucker_mat.clone();
}

cv::Mat MapLine::PluckerMatToCoordinate(const cv::Mat &plucker_mat)
{
    cv::Mat plucker_coord = cv::Mat::zeros(6, 1, CV_32F);

    plucker_coord.at<float>(0) = plucker_mat.at<float>(0, 3);
    plucker_coord.at<float>(1) = plucker_mat.at<float>(1, 3);
    plucker_coord.at<float>(2) = plucker_mat.at<float>(2, 3);

    plucker_coord.at<float>(3) = plucker_mat.at<float>(2, 1); 
    plucker_coord.at<float>(4) = plucker_mat.at<float>(0, 2);
    plucker_coord.at<float>(5) = plucker_mat.at<float>(1, 0);

    return plucker_coord.clone();
}

cv::Mat MapLine::PluckerCoordinateToMat(const cv::Mat &plucker_coord)
{
    cv::Mat nc = plucker_coord.rowRange(0, 3);
    cv::Mat dc = plucker_coord.rowRange(3, 6);

    cv::Mat B0 = cv::Mat::zeros(3, 3, CV_32F);
    B0.at<float>(0, 1) = -nc.at<float>(2);
    B0.at<float>(0, 2) =  nc.at<float>(1);
    B0.at<float>(1, 0) =  nc.at<float>(2);
    B0.at<float>(1, 2) = -nc.at<float>(0);
    B0.at<float>(2, 0) = -nc.at<float>(1);
    B0.at<float>(2, 1) =  nc.at<float>(0);

    cv::Mat B1 = -dc.t();

    cv::Mat plucker_mat = cv::Mat::zeros(4, 4, CV_32F);
    B0.copyTo( plucker_mat.rowRange(0, 3).colRange(0, 3) ); 
    dc.copyTo( plucker_mat.rowRange(0, 3).col(3) );
    B1.copyTo( plucker_mat.row(3).colRange(0, 3) );

    return plucker_mat.clone();
}

cv::Mat MapLine::ProjectToCamera(KeyFrame* kf)
{
    cv::Mat Rcw = kf->GetRotation();
    cv::Mat tcw = kf->GetTranslation();

    cv::Mat tcw_skew_symmetric = cv::Mat::zeros(3, 3, CV_32F);
    tcw_skew_symmetric.at<float>(0, 1) = -tcw.at<float>(2); 
    tcw_skew_symmetric.at<float>(0, 2) =  tcw.at<float>(1); 
    tcw_skew_symmetric.at<float>(1, 0) =  tcw.at<float>(2);
    tcw_skew_symmetric.at<float>(1, 2) = -tcw.at<float>(0);
    tcw_skew_symmetric.at<float>(2, 0) = -tcw.at<float>(1);
    tcw_skew_symmetric.at<float>(2, 1) =  tcw.at<float>(0);

    //normal 投影到相机坐标系
    cv::Mat normal_c = Rcw * this->GetNormal() + tcw_skew_symmetric * Rcw * this->GetDirect();
    //std::cout << normal_c << std::endl;

    // normal 投影到像素平面 ax + by + c = 0
    cv::Mat line_2d = kf->K_inv.t() * normal_c;
    //std::cout << line_2d << std::endl;

    return line_2d.clone ();
}

void MapLine::ComputeOrthByPlucker()
{
    Vector6d plk;
    cv::cv2eigen(mPluckerCoord, plk);

    Vector3d n = plk.head(3);
    Vector3d d = plk.tail(3);

    Vector3d u1 = n/n.norm();
    Vector3d u2 = d/d.norm();
    Vector3d u3 = u1.cross(u2);

    mOrthonormal.at<float>(0) = atan2( u2(2), u3(2) );
    mOrthonormal.at<float>(1) = asin (-u1(2) );
    mOrthonormal.at<float>(2) = atan2( u1(1), u1(0) );

    Vector2d w( n.norm(), d.norm() );
    w = w/w.norm();
    mOrthonormal.at<float>(3) = asin( w(1) );
}

void MapLine::ComputePluckerByOrth()
{
    Vector4d orth;
    cv::cv2eigen(mOrthonormal, orth);

    Vector3d theta = orth.head(3);
    float phi = orth[3];

    float s1 = sin(theta[0]);
    float c1 = cos(theta[0]);
    float s2 = sin(theta[1]);
    float c2 = cos(theta[1]);
    float s3 = sin(theta[2]);
    float c3 = cos(theta[2]);

    Matrix3d R;
    R << c2 * c3, s1 * s2 * c3 - c1 * s3, c1 * s2 * c3 + s1 * s3,
         c2 * s3, s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3,
             -s2,                s1 * c2,                c1 * c2;
    
    float w1 = cos(phi);
    float w2 = sin(phi);

    Vector3d u1 = R.col(0);
    Vector3d u2 = R.col(1);

    Vector3d n = w1 * u1;
    Vector3d d = w2 * u2;

    cv::Mat normal(3, 1, CV_32F);
    cv::Mat direct(3, 1, CV_32F);
    cv::eigen2cv(n, normal); 
    cv::eigen2cv(d, direct);
    normal.copyTo(mPluckerCoord.rowRange(0, 3)); 
    direct.copyTo(mPluckerCoord.rowRange(3, 6));
}
