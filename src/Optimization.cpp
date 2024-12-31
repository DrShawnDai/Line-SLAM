#include "Optimization.h"

Optimizer::Optimizer(Map* pMap)
{
    mpMap = pMap;
}

void Optimizer::onlyLineOpt()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);

    std::vector<KeyFrame*> kfs = mpMap->GetAllKeyFrames();
    const int nKeyFrames = kfs.size();

    std::vector<MapLine*> mls = mpMap->GetAllMapLines();
    const int nMapLines = mls.size();

    double para_Camera[nKeyFrames][7];
    for (int i=0; i<nKeyFrames; i++)
    {
        Vector3d tcw;
        cv::cv2eigen(kfs[i]->GetTranslation(), tcw);
        para_Camera[i][0] = tcw[0];
        para_Camera[i][1] = tcw[1];
        para_Camera[i][2] = tcw[2];

        Matrix3d Rcw;
        cv::cv2eigen(kfs[i]->GetRotation(), Rcw);
        Eigen::Quaterniond q(Rcw);
        para_Camera[i][3] = q.x();
        para_Camera[i][4] = q.y();
        para_Camera[i][5] = q.z();
        para_Camera[i][6] = q.w();

        ceres::Manifold *camera_parameterization = new CameraParameterization();
        problem.AddParameterBlock(para_Camera[i], 7, camera_parameterization);
        problem.SetParameterBlockConstant(para_Camera[i]);
    }

    double para_MapLine[nMapLines][4];
    for (int lineID=0; lineID < nMapLines; lineID++)
    {
        Vector4d lineorth;
        cv::cv2eigen(mls[lineID]->GetOrthonormal(), lineorth);

        para_MapLine[lineID][0] = lineorth[0];
        para_MapLine[lineID][1] = lineorth[1];
        para_MapLine[lineID][2] = lineorth[2];
        para_MapLine[lineID][3] = lineorth[3];

        ceres::Manifold *line_parameterization = new LineOrthParameterization();
        problem.AddParameterBlock( para_MapLine[lineID], 4, line_parameterization);

        for (int camID=0; camID < nKeyFrames; camID++)
        {
            cv::line_descriptor::KeyLine* kl = kfs[camID]->GetKeyLine(lineID);

            // 像素坐标转移到相机归一化坐标，优化中不用再考虑K
            cv::Mat sta_img = (cv::Mat_<float>(2, 1) << kl->startPointX, kl->startPointY);
            cv::Mat end_img = (cv::Mat_<float>(2, 1) << kl->endPointX, kl->endPointY);

            cv::Mat sta_cam = kfs[camID]->pixelToCam(sta_img);
            cv::Mat end_cam = kfs[camID]->pixelToCam(end_img);

            Vector4d obs(sta_cam.at<float>(0), sta_cam.at<float>(1), end_cam.at<float>(0), end_cam.at<float>(1));
            lineProjectionFactor *f = new lineProjectionFactor(obs);
            problem.AddResidualBlock(f, loss_function, para_Camera[camID], para_MapLine[lineID]);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 32;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << std::endl;

    for (int i = 0; i < nMapLines; i++)
    {
        Vector4d orth(para_MapLine[i][0], para_MapLine[i][1], para_MapLine[i][2], para_MapLine[i][3]);
        mpMap->GetMapLine(i)->UpdateByOrth(orth);
    }

    // remove outliers
    // f_manager.removeLineOutlier(Ps, tic, ric);
}

void Optimizer::onlyPointOpt()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);

    std::vector<KeyFrame*> kfs = mpMap->GetAllKeyFrames();
    const int nKeyFrames = kfs.size();

    std::vector<MapPoint*> mps = mpMap->GetAllMapPoints();
    const int nMapPoints = mps.size();

    double para_Camera[nKeyFrames][7];
    for(int i=0; i<nKeyFrames; i++)
    {
        Vector3d tcw;
        cv::cv2eigen(kfs[i]->GetTranslation(), tcw);
        para_Camera[i][0] = tcw[0];
        para_Camera[i][1] = tcw[1];
        para_Camera[i][2] = tcw[2];

        Matrix3d Rcw;
        cv::cv2eigen(kfs[i]->GetRotation(), Rcw);
        Eigen::Quaterniond q(Rcw);
        para_Camera[i][3] = q.x();
        para_Camera[i][4] = q.y();
        para_Camera[i][5] = q.z();
        para_Camera[i][6] = q.w();

        ceres::Manifold *camera_parameterization = new CameraParameterization();
        problem.AddParameterBlock(para_Camera[i], 7, camera_parameterization);
        problem.SetParameterBlockConstant(para_Camera[i]);
    }

    double para_MapPoint[nMapPoints][3];
    for (int ptID=0; ptID < nMapPoints; ptID++)
    {
        Vector3d pt3d;
        cv::cv2eigen(mps[ptID]->GetWorldPos(), pt3d);

        para_MapPoint[ptID][0] = pt3d[0];
        para_MapPoint[ptID][1] = pt3d[1];
        para_MapPoint[ptID][2] = pt3d[2];

        problem.AddParameterBlock(para_MapPoint[ptID], 3);

        for (int camID=0; camID < nKeyFrames; camID++)
        {
            cv::Point* kp = kfs[camID]->GetKeyPoint(ptID);

            // 像素坐标转移到相机归一化坐标，优化中不用再考虑 K
            cv::Mat kp_img = (cv::Mat_<float>(2, 1) << kp->x, kp->y);

            cv::Mat kp_cam = kfs[camID]->pixelToCam(kp_img);

            Vector2d obs(kp_cam.at<float>(0), kp_cam.at<float>(1));
            pointProjectionFactor *f = new pointProjectionFactor(obs);
            problem.AddResidualBlock(f, loss_function, para_Camera[camID], para_MapPoint[ptID]);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 32;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << std::endl;

    for (int i = 0; i < nMapPoints; i++)
    {
        Vector3d pt3d(para_MapPoint[i][0], para_MapPoint[i][1], para_MapPoint[i][2]);
        mpMap->GetMapPoint(i)->UpdateWorldPos(pt3d);
    }

    // remove outliers

}

bool CameraParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d> (delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}

bool CameraParameterization::PlusJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}

bool LineOrthParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    // ref: 2001, Adrien Bartol, Peter Sturm, Structure-From-Motion Using Lines: Representation, Triangulation and Bundle Adjustment
    // theta --> U, phi --> W
    Eigen::Map<const Vector3d> theta(x);
    double phi = *(x + 3);

    double s1 = sin(theta[0]);
    double c1 = cos(theta[0]);
    double s2 = sin(theta[1]);
    double c2 = cos(theta[1]);
    double s3 = sin(theta[2]);
    double c3 = cos(theta[2]);
    Matrix3d R;
    R << c2 * c3, s1 * s2 * c3 - c1 * s3, c1 * s2 * c3 + s1 * s3,
         c2 * s3, s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3,
             -s2,                s1 * c2,                c1 * c2;

    double w1 = cos(phi);
    double w2 = sin(phi);

    // update
    Eigen::Map<const Vector3d> _delta_theta(delta);
    double _delta_phi = *(delta + 3);
    Matrix3d Rz;
    Rz << cos(_delta_theta(2)), -sin(_delta_theta(2)), 0.0,
          sin(_delta_theta(2)),  cos(_delta_theta(2)), 0.0,
                           0.0,                   0.0, 1.0;
    
    Matrix3d Ry;
    Ry << cos(_delta_theta(1)), 0.0, sin(_delta_theta(1)),
                           0.0, 1.0,                  0.0,
         -sin(_delta_theta(1)), 0.0, cos(_delta_theta(1));

    Matrix3d Rx;
    Rx << 1.0,                  0.0,                  0.0,
          0.0, cos(_delta_theta(0)), -sin(_delta_theta(0)),
          0.0, sin(_delta_theta(0)),  cos(_delta_theta(0));
    
    R = R * Rx * Ry * Rz;

    Matrix2d W;
    W << w1, -w2, w2, w1;
    Matrix2d delta_W;
    delta_W << cos(_delta_phi), -sin(_delta_phi), sin(_delta_phi), cos(_delta_phi);
    W = W * delta_W;

    // U' --> theta'. W' --> phi'
    Eigen::Map<Vector3d> theta_plus(x_plus_delta);
    double* phi_plus(x_plus_delta + 3);

    Vector3d u1 = R.col(0);
    Vector3d u2 = R.col(1);
    Vector3d u3 = R.col(2);
    theta_plus[0] = atan2( u2(2), u3(2) );
    theta_plus[1] = asin (-u1(2));
    theta_plus[2] = atan2( u1(1), u1(0) );

    *phi_plus = asin( W(1,0) );

    return true;
}

bool LineOrthParameterization::PlusJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> j(jacobian);
    j.setIdentity();

    return true;
}

lineProjectionFactor::lineProjectionFactor(const Vector4d &_obs_i) : obs_i(_obs_i)
{
    sqrt_info = 460.0 / 1.5 * Eigen::Matrix2d::Identity();
}

// parameters[0]: Tcw
// parameters[1]: line_orth
bool lineProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vector3d tcw(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond quad(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
    Matrix3d Rcw = quad.toRotationMatrix();
    Matrix3d Rwc = Rcw.transpose();
    Vector3d twc = -Rwc * tcw;

    Vector4d line_orth(parameters[1][0], parameters[1][1], parameters[1][2], parameters[1][3]);
    Vector6d line_w = Utility::orth_to_plk(line_orth);

    Vector6d line_c = Utility::plk_world_to_camera(line_w, Rcw, tcw);

    Vector3d nc = line_c.head(3);
    Vector3d dc = line_c.tail(3);
    
    double l_norm = nc(0) * nc(0) + nc(1) * nc(1);
    double l_sqrtnorm = sqrt( l_norm );
    double l_trinorm = l_norm * l_sqrtnorm;

    double e1 = obs_i(0) * nc(0) + obs_i(1) * nc(1) + nc(2);
    double e2 = obs_i(2) * nc(0) + obs_i(3) * nc(1) + nc(2);
    Eigen::Map<Vector2d> residual(residuals);
    residual(0) = e1/l_sqrtnorm;
    residual(1) = e2/l_sqrtnorm;

    residual = sqrt_info * residual;

    if (jacobians)
    {
        // delta_Error / delta_l
        Eigen::Matrix<double, 2, 3> jaco_e_l(2,3);
        jaco_e_l << ( obs_i(0)/l_sqrtnorm - nc(0) * e1 / l_trinorm ), ( obs_i(1)/l_sqrtnorm - nc(1) * e1 / l_trinorm ), 1.0/l_sqrtnorm,
                    ( obs_i(2)/l_sqrtnorm - nc(0) * e2 / l_trinorm ), ( obs_i(3)/l_sqrtnorm - nc(1) * e2 / l_trinorm ), 1.0/l_sqrtnorm;

        jaco_e_l = sqrt_info * jaco_e_l;

        // delta_l / delta_Lc
        Eigen::Matrix<double, 3, 6> jaco_l_Lc(3, 6);
        jaco_l_Lc.setZero();
        jaco_l_Lc.block(0,0,3,3) = Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 2, 6> jaco_e_Lc;
        jaco_e_Lc = jaco_e_l * jaco_l_Lc;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[0]);

            Matrix6d jaco_Lc_ex;
            jaco_Lc_ex.setZero();
            jaco_Lc_ex.block(0,0,3,3) = Rcw * Utility::skew_symmetric(dc);
            jaco_Lc_ex.block(0,3,3,3) = Utility::skew_symmetric( Rcw * (nc + Utility::skew_symmetric(dc) * twc) );
            jaco_Lc_ex.block(3,3,3,3) = Utility::skew_symmetric( Rcw * dc);

            jacobian_ex_pose.leftCols<6>() = jaco_e_Lc * jaco_Lc_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jacobian_lineOrth(jacobians[1]);

            // delta_Lc / delta_Lw
            Matrix6d Hcw;
            Hcw << Rcw, Utility::skew_symmetric(tcw) * Rcw,
                   Eigen::Matrix3d::Zero(),            Rcw;

            // delta_Lw / delta_orth
            Vector3d nw = line_w.head(3);
            Vector3d vw = line_w.tail(3);
            Vector3d u1 = nw/nw.norm();
            Vector3d u2 = vw/vw.norm();
            Vector3d u3 = u1.cross(u2);
            Vector2d w( nw.norm(), vw.norm() );
            w = w/w.norm();

            Eigen::Matrix<double, 6, 4> jaco_Lw_orth;
            jaco_Lw_orth.setZero();
            jaco_Lw_orth.block(3,0,3,1) =  w(1) * u3;
            jaco_Lw_orth.block(0,1,3,1) = -w(0) * u3;
            jaco_Lw_orth.block(0,2,3,1) =  w(0) * u2;
            jaco_Lw_orth.block(3,2,3,1) = -w(1) * u1;
            jaco_Lw_orth.block(0,3,3,1) = -w(1) * u1;
            jaco_Lw_orth.block(3,3,3,1) =  w(0) * u2;

            jacobian_lineOrth = jaco_e_Lc * Hcw * jaco_Lw_orth;
        }
    }

    return true;
}

pointProjectionFactor::pointProjectionFactor(const Vector2d &_obs_i): obs_i(_obs_i)
{
    sqrt_info = 460.0 / 1.5 * Eigen::Matrix2d::Identity();
}

// parameters[0]: Tcw
// parameters[1]: Pw
bool pointProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vector3d tcw(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond quad(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
    Matrix3d Rcw = quad.toRotationMatrix();
    Matrix3d Rwc = Rcw.transpose();
    Vector3d twc = -Rwc * tcw;

    Vector3d Pw( parameters[1][0], parameters[1][1], parameters[1][2]);

    Vector3d Pc = Rcw * Pw + tcw;
    Vector2d pc(Pc(0)/Pc(2), Pc(1)/Pc(2));

    Eigen::Map<Vector2d> residual(residuals);
    residual = obs_i - pc;
    residual = sqrt_info * residual;

    if (jacobians)
    {
        // delta_Error / delta_Pc
        Eigen::Matrix<double, 2, 3> jaco_e_Pc(2, 3);
        jaco_e_Pc << -1.0/Pc(2),        0.0, Pc(0)/(Pc(2) * Pc(2)),
                            0.0, -1.0/Pc(2), Pc(1)/(Pc(2) * Pc(2));
        
        if(jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_cam(jacobians[0]);

            // Matrix6d jaco_Lc_ex;
            // jaco_Lc_ex.setZero();
            // jaco_Lc_ex.block(0,0,3,3) = Rcw * Utility::skew_symmetric(dc);
            // jaco_Lc_ex.block(0,3,3,3) = Utility::skew_symmetric(Rcw * (nc + Utility::skew_symmetric(dc) * twc));
            // jaco_Lc_ex.block(3,3,3,3) = Utility::skew_symmetric(Rcw * dc);

            // jacobian_ex_pose.leftCols<6>()= jaco_e_Lc * jaco_Lc_ex;
            // jacobian_ex_pose.rightCols<1>().setZero();
        }

        if(jacobians[1])
        {
            // delta_Pc / delta_Pw
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian_Pw(jacobians[1]);

            jacobian_Pw = jaco_e_Pc * Rcw;
        }
    }

    return true;
}