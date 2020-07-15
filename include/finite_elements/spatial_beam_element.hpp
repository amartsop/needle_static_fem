#pragma once 

#include <iostream>
#include <armadillo>
#include "axial_element.hpp"
#include "bending_element.hpp"


class SpatialBeamElement
{

public:
    
    SpatialBeamElement(double length, double area, double inertia_y, 
        double inertia_z, double young_modulus, double density);
    
    arma::dmat get_mass_matrix(void) { return m_mass; }
    arma::dmat get_stiffness_matrix(void) { return m_stiffness; }

private:

    double m_element_length; // (m)
    double m_element_area; // (m^2)
    double m_element_inertia_y; // (kg * m^4)
    double m_element_inertia_z; // (kg * m^4)
    double m_element_young_modulus; // (N / m^2)
    double m_element_density; // (kg / m^3)
    
    // Number of compoments 
    static const int m_n = 10; 

    // Bending element
    arma::dmat m_mass; // Bending element mass matrix
    arma::dmat m_stiffness; // Bending element stiffness matrix

public:
    // Locator matrix
    static arma::dmat locator_matrix(arma::ivec locator_vector, 
        uint32_t n_cols);

    // Shape function
    static arma::dmat shape_function(double x, double l);
};

SpatialBeamElement::SpatialBeamElement(double length, double area, double inertia_y, 
        double inertia_z, double young_modulus, double density)
{
    m_element_length = length;
    m_element_area = area;
    m_element_inertia_y = inertia_y;
    m_element_inertia_z = inertia_z;
    m_element_young_modulus = young_modulus;
    m_element_density = density;

    // Axial element
    auto axial = AxialElement(m_element_length, m_element_area, 
        m_element_young_modulus, m_element_density);

    arma::dmat m_u = axial.get_mass_matrix();
    arma::dmat k_u = axial.get_stiffness_matrix();

    arma::ivec lu = {1, 6}; 
    arma::dmat lu_mat = locator_matrix(lu, m_n);
    
    // Bending element y axis 
    auto bending_y = BendingElement(m_element_length, m_element_area, 
        m_element_inertia_z, m_element_young_modulus, m_element_density);

    arma::dmat m_v = bending_y.get_mass_matrix();
    arma::dmat k_v = bending_y.get_stiffness_matrix();

    arma::ivec lv = {2, 4, 7, 9};
    arma::dmat lv_mat = locator_matrix(lv, m_n);
    
    // Bending element z axis 
    auto bending_z = BendingElement(m_element_length, m_element_area, 
        m_element_inertia_y, m_element_young_modulus, m_element_density);

    // Transformation matrix 
    arma::dmat tr = {{1, -1, 1, -1}, {-1, 1, -1, 1}, {1, -1, 1, -1}, {-1, 1, -1, 1}};
    
    arma::dmat m_w = tr % bending_z.get_mass_matrix();
    arma::dmat k_w = tr % bending_z.get_stiffness_matrix();

    arma::ivec lw = {3, 5, 8, 10}; 
    arma::dmat lw_mat = locator_matrix(lw, m_n);

    // Spatial beam mass matrix
    m_mass = lu_mat.t() * m_u * lu_mat  +lv_mat.t() * m_v * lv_mat + 
        lw_mat.t() * m_w * lw_mat;

    // Spatial beam stiffness matrix
    m_stiffness = lu_mat.t() * k_u * lu_mat + lv_mat.t() * k_v * lv_mat + 
        lw_mat.t() * k_w * lw_mat;
}

arma::dmat SpatialBeamElement::locator_matrix(arma::ivec locator_vector, 
    uint32_t n_cols)
{
    arma::dmat l_mat = arma::zeros<arma::dmat>(locator_vector.n_rows, n_cols);
    
    for(uint32_t i = 0; i < locator_vector.n_rows; i++)    
    {
        l_mat(i, locator_vector(i) - 1) = 1;
    }
    return l_mat;
}

arma::dmat SpatialBeamElement::shape_function(double x, double l)
{

    arma::ivec lu = {1, 6}; arma::dmat lu_mat = locator_matrix(lu, m_n);
    arma::ivec lv = {2, 4, 7, 9}; arma::dmat lv_mat = locator_matrix(lv, m_n);
    arma::ivec lw = {3, 5, 8, 10}; arma::dmat lw_mat = locator_matrix(lw, m_n);
    
    arma::dmat phi_u = AxialElement::shape_function(x, l);
    arma::dmat phi_v = BendingElement::shape_function(x, l);
    arma::dmat phi_w = BendingElement::shape_function(x, l);
    arma::dmat transfn = {1, -1, 1, -1};
    phi_w = phi_w % transfn;

    arma::dmat phi = arma::join_vert(phi_u * lu_mat, phi_v * lv_mat,
        phi_w * lw_mat);

    return phi;
}