#pragma once 

#include <iostream>
#include <vector>
#include <armadillo>

#include "boundary_conditions.hpp"
#include "input_coordinates.hpp"
#include "forces.hpp"

class LinearFemModel
{
public:

    LinearFemModel(BoundaryConditions *bcs, InputCoordinates *input_coords,
        Forces *forces);

    arma::dvec calculate(arma::dvec state, double t);

    arma::dvec get_reaction_forces(void) {return m_qg_force; }

    uint get_model_size(void) { return m_size; }

private:

    // Boundary condition handle
    BoundaryConditions *m_bcs;

    // Input coordinates handle 
    InputCoordinates *m_input_coords;
    
    // Force handle 
    Forces *m_forces;

    // State vector size 
    uint m_size; 

    // Mass matrices 
    arma::dmat m_maa, m_mag, m_mgg, m_mga, m_maa_inv;
 
    // Stiffness matrices 
    arma::dmat m_kaa, m_kag, m_kgg, m_kga;

    // Reaction forces
    arma::dvec m_qg_force;
};

LinearFemModel::LinearFemModel(BoundaryConditions *bcs, 
    InputCoordinates *input_coords, Forces *forces)
{
    // Boundary conditions handle
    m_bcs = bcs;

    // Input coordinates handle
    m_input_coords = input_coords;

    // Input coordinates handle
    m_forces = forces;
    
    // State vector size 
    arma::ivec la = m_bcs->get_uknown_dofs_id();
    m_size = 2 * la.n_rows;

    // Mass Matrices 
    m_maa = m_bcs->get_mass_matrix_maa(); m_mag = m_bcs->get_mass_matrix_mag();
    m_mgg = m_bcs->get_mass_matrix_mgg(); m_mga = m_bcs->get_mass_matrix_mga();
    m_maa_inv = arma::pinv<arma::dmat>(m_maa);
    // m_maa_inv.eye(m_maa.n_rows, m_maa.n_cols);

    // Stiffness Matrices 
    m_kaa = m_bcs->get_stiffness_matrix_kaa();
    m_kag = m_bcs->get_stiffness_matrix_kag();
    m_kgg = m_bcs->get_stiffness_matrix_kgg();
    m_kga = m_bcs->get_stiffness_matrix_kga();
}


arma::dvec LinearFemModel::calculate(arma::dvec state, double t)
{
    // Size 
    uint xi_size = m_size  / 2;

    // Known coordinates trajectory 
    m_input_coords->update(state, t);
    arma::dvec qg = m_input_coords->get_displacement_qg();
    arma::dvec qg_ddot = m_input_coords->get_acceleration_qg_ddot();

    // Uknown coordinates
    arma::dvec qa = state.rows(0, xi_size - 1);
    arma::dvec qa_dot = state.rows(xi_size, state.n_rows - 1);

    // Forces
    arma::dvec fa = m_forces->get_force_fa(state, t);

    // State 
    arma::dvec x1 = qa;
    arma::dvec x2 = qa_dot;

    // Reaction forces
    arma::dvec qa_ddot = arma::solve(m_maa, fa - m_kaa * x1
        - m_mag * qg_ddot - m_kag * qg, arma::solve_opts::likely_sympd);

    m_qg_force = m_mgg * qg_ddot + m_mga * qa_ddot + m_kgg * qg + m_kga * qa;

    // x1_dot
    arma::dvec x1_dot = x2;
    
    // x2_dot 
    arma::dvec x2_dot = qa_ddot;

    // x_dot
    arma::dvec x_dot = arma::join_vert(x1_dot, x2_dot);

    return x_dot;

}
