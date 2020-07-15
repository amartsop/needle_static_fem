#pragma once 

#include <iostream>
#include <armadillo>

class TorsionalElement
{

public:

    TorsionalElement(double length, double polar_inertia, double shear_modulus,
        double torsional_constant, double density);

    arma::dmat get_mass_matrix(void) { return m_mass; }
    arma::dmat get_stiffness_matrix(void) { return m_stiffness; }
    
    static arma::dmat shape_function(double x, double l);

private:
    double m_element_length; // (m)
    double m_element_polar_inertia; // (m^4)
    double m_element_shear_modulus; // (N / m^2)
    double m_element_torsional_constant; // (m^4)
    double m_element_density; // (kg / m^3)

    // Torsional element
    arma::dmat m_mass; // Torsional element mass matrix
    arma::dmat m_stiffness; // Torsional element stiffness matrix
};

TorsionalElement::TorsionalElement(double length, double polar_inertia, 
        double shear_modulus, double torsional_constant, double density)
{
    m_element_length = length;
    m_element_polar_inertia = polar_inertia;
    m_element_shear_modulus = shear_modulus;
    m_element_torsional_constant = torsional_constant;
    m_element_density = density;

    /** Torsional element **/
    // Mass matrix 
    double mass_coef = (1.0 / 6.0) * (m_element_density * 
        m_element_polar_inertia * m_element_length);
    m_mass = {{2, 1}, {1, 2}}; m_mass *= mass_coef;

    // Stifness matrix 
    double stif_coef = (m_element_shear_modulus * m_element_torsional_constant) / m_element_length;
    m_stiffness = {{1, -1}, {-1, 1}}; m_stiffness *= stif_coef;

}

arma::dmat TorsionalElement::shape_function(double x, double l)
{
    double ksi = x / l; arma::dmat phi_mat = {1 - ksi, ksi};
    return phi_mat;
}