#pragma once

#include <armadillo>
#include "boundary_conditions.hpp"

class InputCoordinates
{

public:
    InputCoordinates(BoundaryConditions *bcs);

    // Update input coordinates 
    void update(arma::dvec q, double t);

    arma::dvec get_displacement_qg(void) { return m_qg; }
    arma::dvec get_velocity_qg_dot(void){ return m_qg_dot; }
    arma::dvec get_acceleration_qg_ddot(void) {  return m_qg_ddot; }

private:
    
    // Boundary conditions handle
    BoundaryConditions *m_bcs; 
    
    //Known dofs 
    arma::ivec m_lg;

    // Displacement function per dof
    double displacement_per_dof(arma::dvec q, double t, uint dof_id);
    double velocity_per_dof(arma::dvec q, double t, uint dof_id);
    double acceleration_per_dof(arma::dvec q, double t, uint dof_id);

    // Displacement known coordinates 
    arma::dvec m_qg;
    
    // Velocity known coordinates 
    arma::dvec m_qg_dot;
    
    // Acceleration known coordinates 
    arma::dvec m_qg_ddot;
};

InputCoordinates::InputCoordinates(BoundaryConditions *bcs)
{
    // Boundary conditions
    m_bcs = bcs;
    
    // Known dofs 
    m_lg = m_bcs->get_known_dofs_id();

    // Displacement initialization
    m_qg = arma::zeros<arma::dvec>(m_lg.n_rows);

    // Velocity initialization
    m_qg_dot = arma::zeros<arma::dvec>(m_lg.n_rows);
    
    // Acceleration initialization
    m_qg_ddot = arma::zeros<arma::dvec>(m_lg.n_rows);
}


void InputCoordinates::update(arma::dvec q, double t)
{
    for (uint i = 0; i < m_lg.n_rows; i++)
    {
        m_qg(i) = displacement_per_dof(q, t, m_lg(i));
        m_qg_dot(i) = velocity_per_dof(q, t, m_lg(i));
        m_qg_ddot(i) = acceleration_per_dof(q, t, m_lg(i));
    }
}

double InputCoordinates::displacement_per_dof(arma::dvec q, double t, uint dof_id)
{
    return 0;
}
   
double InputCoordinates::velocity_per_dof(arma::dvec q, double t, uint dof_id)
{
    return 0;
}

double InputCoordinates::acceleration_per_dof(arma::dvec q, double t, uint dof_id)
{
    return 0;
}