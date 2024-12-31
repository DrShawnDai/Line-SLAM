#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include "Utility.h" 
#include "Map.h"

#include <ceres/ceres.h>

class CameraParameterization : public ceres::Manifold
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const; 
    virtual bool PlusJacobian(const double *x, double *jacobian) const;
    virtual bool Minus(const double* y, const double* x, double* y_minus_x) const {}
    virtual bool MinusJacobian(const double* x, double* jacobian) const {}
    virtual int AmbientSize() const { return 7; }
    virtual int TangentSize() const { return 6; }
};

class LineOrthParameterization: public ceres::Manifold
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool PlusJacobian(const double *x, double *jacobian) const;
    virtual bool Minus(const double* y, const double* x, double* y_minus_x) const {}
    virtual bool MinusJacobian (const double* x, double* jacobian) const {}
    virtual int AmbientSize() const { return 4; }; 
    virtual int TangentSize() const { return 4; };
};

class lineProjectionFactor : public ceres::SizedCostFunction<2, 7, 4>
{ 
public:
    lineProjectionFactor(const Vector4d &_obs_i);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    Vector4d obs_i;
    Matrix2d sqrt_info;
};

class pointProjectionFactor : public ceres::SizedCostFunction<2, 7, 3>
{ 
public:
    pointProjectionFactor(const Vector2d &obs_i);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    Vector2d obs_i;
    Matrix2d sqrt_info;
};

class Optimizer
{ 
public:
    Optimizer(Map* pMap);

    // 固定相机pose，只优化 Mapline 的参数
    void onlyLineOpt();

    // 固定相机pose，只优化 MapPoint 的参数
    void onlyPointOpt();

private:
    Map* mpMap;
};

#endif // OPTIMIZATION_H