#pragma once 

#include <iostream>
#include <armadillo>
#include <vector>

#include "euler_rotations.hpp"
#include "spatial_beam.hpp"
#include "gnuplot-iostream.h"

class PostProcessing
{
public:
    // Postprocessing for plotting and analysis
    PostProcessing(SpatialBeam *needle);
    
    // Get the beam coordinates in the floating frame of reference
    arma::dmat get_beam_coordinates_ffr(arma::dmat elastic_coordinates);

    // Get the beam coordinates in the inertial frame of reference
    arma::dmat get_beam_coordinates_inertial(arma::dvec roc_g_g, 
        arma::dvec euler_angles, arma::dvec elastic_coordinates) ;

private:

    // Graphs handle
    Gnuplot m_gp;

    // Needle handle
    SpatialBeam *m_needle;

    // Number of points in each element
    const int npel = 50;     
};

PostProcessing::PostProcessing(SpatialBeam *needle)
{
    // Needle handle
    m_needle = needle;
}


arma::dmat PostProcessing::get_beam_coordinates_ffr(arma::dmat elastic_coordinates)
{

    // Beam points 
    std::vector<double> m_beam_points_x, m_beam_points_y, m_beam_points_z;

    // Beam points armadillo 
    arma::dvec rx, ry, rz;
    
    // Beam length
    auto l = m_needle->get_needle_length();

    // Element length
    auto le = m_needle->get_elements_length();

    // Elements num 
    auto elements_num = m_needle->get_elements_num();

    // Dofs num 
    auto dofs_num = m_needle->get_elements_dof();

    // Discretize element
    arma::dvec x2j_vec = arma::linspace<arma::dvec> (0, le, npel);

    for (int j = 1; j <= elements_num; j++)
    {
        arma::ivec l_vec_j = arma::regspace<arma::ivec>(5 * j - 4, 1, 
            5 * j + 5);
        
        arma::dmat lj_mat = SpatialBeam::locator_matrix(l_vec_j, dofs_num);

        // Element local coordinates 
        arma::dvec e_j =  lj_mat * elastic_coordinates;
       
        for (int i = 0; i < npel; i++) 
        { 
            //Shape function for element j
            arma::dmat phi_mat_j = SpatialBeam::shape_function(x2j_vec(i), le);

            // Deformation of point pj
            arma::dvec rp0pj_f_f = phi_mat_j * e_j;

            // Initial position of pj wrt to f origin
            arma::dvec rap0j_f_f = { (double) (j - 1) * le + x2j_vec(i), 0.0f, 0.0f};
            
            // Final position of pj wrt to f origin
            arma::dvec rapj_f_f = rap0j_f_f + rp0pj_f_f;
    
            // Final position of pj wrt to f origin
            m_beam_points_x.push_back(rapj_f_f(0));
            m_beam_points_y.push_back(rapj_f_f(1));
            m_beam_points_z.push_back(rapj_f_f(2));
        }
    }

    // Armadillo matrices of beam elements 
    rx = arma::vec(m_beam_points_x); ry = arma::vec(m_beam_points_y);
    rz = arma::vec(m_beam_points_z);

    return (arma::join_horiz(rx, ry, rz));
}

arma::dmat PostProcessing::get_beam_coordinates_inertial(arma::dvec roa_g_g, 
    arma::dvec euler_angles, arma::dvec elastic_coordinates)
{
    // Rigid body coordinates
    arma::dmat rot_f_g = EulerRotations::rotation(euler_angles);
    
    // Beam points 
    std::vector<double> m_beam_points_x, m_beam_points_y, m_beam_points_z;

    // Beam points armadillo 
    arma::dvec rx, ry, rz;
    
    // Beam length
    auto l = m_needle->get_needle_length();

    // Element length
    auto le = m_needle->get_elements_length();

    // Elements num 
    auto elements_num = m_needle->get_elements_num();

    // Dofs num 
    auto dofs_num = m_needle->get_elements_dof();

    // Discretize element
    arma::dvec x2j_vec = arma::linspace<arma::dvec> (0, le, npel);

    for (int j = 1; j <= elements_num; j++)
    {
        arma::ivec l_vec_j = arma::regspace<arma::ivec>(5 * j - 4, 1, 
            5 * j + 5);
        
        arma::dmat lj_mat = SpatialBeam::locator_matrix(l_vec_j, dofs_num);

        // Element local coordinates 
        arma::dvec e_j =  lj_mat * elastic_coordinates;
       
        for (int i = 0; i < npel; i++) 
        { 
            //Shape function for element j
            arma::dmat phi_mat_j = SpatialBeam::shape_function(x2j_vec(i), le);

            // Deformation of point pj
            arma::dvec rp0pj_f_f = phi_mat_j * e_j;

            // Initial position of pj wrt to f origin
            arma::dvec rap0j_f_f = { (double) (j - 1) * le + x2j_vec(i), 0.0f, 0.0f};
            
            // Final position of pj wrt to f origin
            arma::dvec rapj_f_f = rap0j_f_f + rp0pj_f_f;

            // Final position of pj wrt to G origin 
            arma::dvec ropj_g_g = roa_g_g + rot_f_g *  rapj_f_f;
            
            // Final position of pj wrt to f origin
            m_beam_points_x.push_back(ropj_g_g(0));
            m_beam_points_y.push_back(ropj_g_g(1));
            m_beam_points_z.push_back(ropj_g_g(2));
        }
    }

    // Armadillo matrices of beam elements 
    rx = arma::vec(m_beam_points_x); ry = arma::vec(m_beam_points_y);
    rz = arma::vec(m_beam_points_z);

    return (arma::join_horiz(rx, ry, rz));
}