#pragma once

#include <iostream>
#include <armadillo>

class AxialElement
{

public:

    AxialElement(double length, double area, double young_modulus,
        double density);

    arma::dmat get_mass_matrix(void) { return m_mass; }
    arma::dmat get_stiffness_matrix(void) { return m_stiffness; }

    static arma::dmat shape_function(double x, double l);

private:
    double m_element_length; // (m)
    double m_element_area; // (m^2)
    double m_element_young_modulus; // (N / m^2)
    double m_element_density; // (kg / m^3)

    // Axial element
    arma::dmat m_mass; // Axial  element mass matrix
    arma::dmat m_stiffness; // Axial element stiffness matrix
};

AxialElement::AxialElement(double length, double area, 
    double young_modulus, double density)
{
    m_element_length = length;
    m_element_area = area;
    m_element_young_modulus = young_modulus;
    m_element_density = density;

    /** Axial element **/
    // Mass matrix 
    double mass_coef = (1.0 / 6.0) * (m_element_density * m_element_area * m_element_length);
    m_mass = {{2, 1}, {1, 2}}; m_mass *= mass_coef;

    // Stiffness matrix 
    double stif_coef = (m_element_young_modulus * m_element_area) / m_element_length;
    m_stiffness = {{1, -1}, {-1, 1}}; m_stiffness *= stif_coef;
}

arma::dmat AxialElement::shape_function(double x, double l)
{
    double ksi = x / l; arma::dmat phi_mat = {1 - ksi, ksi};
    return phi_mat;
}