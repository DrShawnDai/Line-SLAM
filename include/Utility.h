#ifndef UTILITY_H 
#define UTILITY_H

#include <Eigen/Dense>

typedef Eigen::Matrix<double, 2, 1> Vector2d;
typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef Eigen::Matrix<double, 4, 1> Vector4d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

typedef Eigen::Matrix<double, 2, 2> Matrix2d;
typedef Eigen::Matrix<double, 3, 3> Matrix3d;
typedef Eigen::Matrix<double, 3, 3> Matrix4d; 
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

class Utility
{ 
public:
    static Vector4d plk_to_orth(Vector6d plk); 
    static Vector6d orth_to_plk(Vector4d orth);
    static Vector6d plk_world_to_camera(Vector6d plk_w, Matrix3d Rcw, Vector3d tcw);
    static Matrix3d skew_symmetric(Vector3d v);

    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
    {
        typedef typename Derived::Scalar Scalar_t;

        Eigen::Quaternion<Scalar_t> dq;
        Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
        half_theta /= static_cast<Scalar_t>(2.0);
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }
};

#endif // UTILITY_H