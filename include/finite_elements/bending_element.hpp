#pragma once 

#include <iostream>
#include <armadillo>

class BendingElement
{

public:

    BendingElement(double length, double area, double inertia, 
        double young_modulus, double density);

    arma::dmat get_mass_matrix(void) { return m_mass; }
    arma::dmat get_stiffness_matrix(void) { return m_stiffness; }

    static arma::dmat shape_function(double x, double l);

private:

    double m_element_length; // (m)
    double m_element_area; // (m^2)
    double m_element_inertia; // (kg * m^2)
    double m_element_young_modulus; // (N / m^2)
    double m_element_density; // (kg / m^3)
    
    // Bending element
    arma::dmat m_mass; // Bending element mass matrix
    arma::dmat m_stiffness; // Bending element stiffness matrix
};

BendingElement::BendingElement(double length, double area, double inertia, 
        double young_modulus, double density)
{
    m_element_length = length;
    m_element_area = area;
    m_element_inertia = inertia;
    m_element_young_modulus = young_modulus;
    m_element_density = density;

    /** Bending element **/
    // Mass matrix 
    double mass_coef = (1.0 / 420.0) * (m_element_density * m_element_area * 
        m_element_length);

    m_mass = {{156, 22 * m_element_length, 54, -13 * m_element_length}, 
        {22 * m_element_length, 4 * pow(m_element_length, 2.0), 13 * 
        m_element_length, -3 * pow(m_element_length, 2.0)},
        {54, 13 * m_element_length, 156, -22 * m_element_length}, 
        {-13 * m_element_length, -3 * pow(m_element_length, 2.0), 
        -22 * m_element_length, 4 * pow(m_element_length, 2.0)}};
    
    m_mass *= mass_coef;


    // Stifness matrix 
    double stif_coef = (m_element_young_modulus * m_element_inertia) / 
        pow(m_element_length, 3.0);

    m_stiffness ={{12, 6 * m_element_length, -12, 6 * m_element_length},
        {6 * m_element_length, 4 * pow(m_element_length, 2.0), 
        -6 * m_element_length, 2 * pow(m_element_length, 2.0)},
        {-12, -6 * m_element_length, 12, -6 * m_element_length},
        {6 * m_element_length, 2 * pow(m_element_length, 2.0), 
        -6 * m_element_length, 4 * pow(m_element_length, 2.0)}};
    
    m_stiffness *= stif_coef;

    sleep(1);
}

arma::dmat BendingElement::shape_function(double x, double l)
{
    double ksi = x / l;
    arma::dmat phi_mat = { 
        1 - 3 * pow(ksi, 2.0)  + 2 * pow(ksi, 3.0), 
        l * (ksi - 2 * pow(ksi, 2.0) + pow(ksi, 3.0)), 
        3 * pow(ksi, 2.0) - 2 * pow(ksi, 3.0), 
        l * (pow(ksi, 3.0) - pow(ksi, 2.0))};

    return phi_mat;
}