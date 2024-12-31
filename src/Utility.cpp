#include "Utility.h"

Vector4d Utility::plk_to_orth(Vector6d plk)
{
    Vector4d orth;
    Vector3d n = plk.head(3);
    Vector3d v = plk.tail(3);

    Vector3d u1 = n/n.norm();
    Vector3d u2 = v/v.norm();
    Vector3d u3 = u1.cross(u2);

    // todo:: use SO3
    orth[0] = atan2( u2(2), u3(2) );
    orth[1] = asin ( -u1(2) );
    orth[2] = atan2( u1(1), u1(0) );

    Vector2d w( n.norm(), v.norm() ); 
    w= w/w.norm();
    orth[3] = asin( w(1) );

    return orth;
}

Vector6d Utility::orth_to_plk(Vector4d orth)
{
    Vector6d plk;

    Vector3d theta = orth.head(3);
    double phi = orth[3];

    double s1 = sin(theta[0]);
    double c1 = cos(theta[0]);
    double s2 = sin(theta[1]);
    double c2 = cos(theta[1]);
    double s3 = sin(theta[2]);
    double c3 = cos(theta[2]);

    Eigen::Matrix3d R;
    R << c2 * c3, s1 * s2 * c3 - c1 * s3, c1 * s2 * c3 + s1 * s3,
         c2 * s3, s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3,
             -s2,                s1 * c2,                c1 * c2;

    double w1 = cos(phi);
    double w2 = sin(phi);

    Vector3d u1 = R.col(0);
    Vector3d u2 = R.col(1);

    Vector3d n = w1 * u1;
    Vector3d v = w2 * u2;

    plk.head(3) = n;
    plk.tail(3) = v;

    return plk;
}

Vector6d Utility::plk_world_to_camera(Vector6d plk_w, Matrix3d Rcw, Vector3d tcw)
{
    Vector3d nw = plk_w.head(3);
    Vector3d vw = plk_w.tail(3);

    Vector3d nc = Rcw * nw + skew_symmetric(tcw) * Rcw * vw;
    Vector3d vc = Rcw * vw;

    Vector6d plk_c;
    plk_c.head(3) = nc;
    plk_c.tail(3) = vc;
    return plk_c;
}

Matrix3d Utility::skew_symmetric(Vector3d v)
{
    Matrix3d S;
    S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return S;
}