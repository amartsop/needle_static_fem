#pragma once 

#include <iostream>
#include <armadillo>
#include "needle_properties.hpp"
#include "finite_elements/spatial_beam_element.hpp"

class SpatialBeam : public NeedleProperties
{
public:

    SpatialBeam(uint elements, uint elements_dof);
    arma::dmat get_mass_matrix() { return m_mass; }
    arma::dmat get_stiffness_matrix() { return m_stiffness; }

    static arma::dmat locator_matrix(arma::ivec locator_vector, uint32_t n_cols);
    static arma::dmat shape_function(double x, double l);
    
    // Getters 
    uint get_elements_num() { return m_elements; }
    uint get_elements_dof() { return m_elements_dof; }
    double get_elements_length() { return m_element_length; }

private:

    // Number of elements 
    uint m_elements;

    // Number of degrees of freedom
    uint m_elements_dof;

    // Element lenght (m)
    double m_element_length;

    // Element radius (m)
    double m_element_radius;

    // Element cross-sectional area (m^2)
    double m_element_area;

    // Element area moment of inertia (m^4)
    double m_iyy, m_izz; 

    // Element area polar moment of inertia (m^4)
    double m_ip, m_j;
    
    // Beam mass matrix 
    arma::dmat m_mass;
    
    // Beam stiffness matrix 
    arma::dmat m_stiffness;

};


SpatialBeam::SpatialBeam(uint elements, uint elements_dof)
{
    // Number of elements
    m_elements = elements;

    // Number of degrees of freedom
    m_elements_dof = elements_dof;

    // Element length
    m_element_length = m_needle_length / (double) m_elements;

    // Element radius
    m_element_radius = m_needle_radius;
    
    // Element cross-sectional area (m^2)
    m_element_area = M_PI * pow(m_element_radius, 2.0);

    // Element area moment of inertia (m^4)
    m_iyy = (M_PI / 4) * pow(m_needle_radius, 4.0);
    m_izz = (M_PI / 4) * pow(m_needle_radius, 4.0);
    
    // Element area polar moment of inertia (m^4)
    m_ip = M_PI * pow(m_element_radius, 4.0) / 2.0; m_j = m_ip; 
    
    // Spatial beam element 
    auto spatial_beam = SpatialBeamElement(m_element_length, m_element_area,
        m_iyy, m_izz, m_needle_young_modulus, m_needle_density);


    arma::dmat element_mass  = spatial_beam.get_mass_matrix();
    arma::dmat element_stiffness = spatial_beam.get_stiffness_matrix();

    // Global structure matrices 
    m_mass = arma::zeros<arma::dmat>(5 * m_elements + 5, 5 * m_elements + 5);
    m_stiffness = arma::zeros<arma::dmat>(5 * m_elements + 5, 5 * m_elements + 5);

    for (uint i = 0; i < m_elements; i++)
    { 
        uint j = i + 1;
        arma::ivec l_vec_j = arma::regspace<arma::ivec>(5 * j - 4, 1, 
            5 * j + 5);
        
        arma::dmat l_mat_j = locator_matrix(l_vec_j, 5 * m_elements + 5);
        
        m_mass += l_mat_j.t() * element_mass * l_mat_j; 
        m_stiffness += l_mat_j.t() * element_stiffness * l_mat_j; 
    }

}

arma::dmat SpatialBeam::locator_matrix(arma::ivec locator_vector, 
    uint n_cols)
{
    arma::dmat l_mat = SpatialBeamElement::locator_matrix(locator_vector,
        n_cols);
    return l_mat;    
}


arma::dmat SpatialBeam::shape_function(double x, double l)
{
    arma::dmat phi = SpatialBeamElement::shape_function(x, l);
    return phi;
}