#ifndef EULER_ROTATIONS_H
#define EULER_ROTATIONS_H

#include <iostream>
#include <armadillo>

class EulerRotations
{
public:
    
    EulerRotations();
    static arma::dmat basic_rotation_x(double x);
    static arma::dmat basic_rotation_y(double x);
    static arma::dmat basic_rotation_z(double x);
    static arma::dmat rotation(double phi, double theta, double psi);
    static arma::dmat rotation(arma::dvec euler_angles);

private:

};


EulerRotations::EulerRotations()
{
}

arma::dmat EulerRotations::basic_rotation_x(double x)
{
    arma::dmat rot = {
        {1.0f, 0.0f, 0.0f}, 
        {0.0f, cos(x), -sin(x)}, 
        {0.0f, sin(x), cos(x)}};
    return rot;
}

arma::dmat EulerRotations::basic_rotation_y(double x)
{
    arma::dmat rot = {
        {cos(x), 0.0f, sin(x)}, 
        {0.0f, 1.0f, 0.0f}, 
        {-sin(x), 0.0f, cos(x)}};
    return rot;
}

arma::dmat EulerRotations::basic_rotation_z(double x)
{
    arma::dmat rot = {
        {cos(x), -sin(x), 0.0f}, 
        {sin(x), cos(x), 0.0f}, 
        {0.0f, 0.0f, 1.0f}};
    return rot;
}

arma::dmat EulerRotations::rotation(double phi, double theta, double psi)
{

    arma::dmat rotx =  basic_rotation_x(phi);
    arma::dmat roty = basic_rotation_y(theta);
    arma::dmat rotz = basic_rotation_z(psi);

    arma::dmat rot = (rotz * roty * rotx);
    return rot;
}

arma::dmat EulerRotations::rotation(arma::dvec euler_angles)
{
    return rotation(euler_angles(0), euler_angles(1), euler_angles(2));
}

#endif