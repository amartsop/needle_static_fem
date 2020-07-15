#pragma once 

#include <iostream>
#include "boundary_conditions.hpp"


class Forces
{
public:

    Forces(BoundaryConditions *bcs);
    arma::dvec get_force_fa(arma::dvec q, double t);

private:

    arma::dvec force_per_node(arma::dvec q, double t, uint node_id);
    arma::dvec moment_per_node(arma::dvec q, double t, uint node_id);

    void set_point_loading(arma::dvec q, double t, uint node_id);
   
    // Boundary conditions handle
    BoundaryConditions *m_bcs; 

    // Forces 
    arma::dvec m_fa;

    //Uknown dofs 
    arma::ivec m_la;
};


Forces::Forces(BoundaryConditions *bcs)
{
    // Boundary conditions
    m_bcs = bcs;

    // Uknown dofs 
    m_la = m_bcs->get_uknown_dofs_id();

    // Forces
    m_fa = arma::zeros<arma::dvec>(m_la.n_rows);
}

arma::dvec Forces::get_force_fa(arma::dvec q, double t)
{
    uint nodes_num = m_bcs->get_nodes_num();
    arma::ivec forces_nodes_id = {nodes_num};

    for (uint i = 0; i < forces_nodes_id.n_rows; i++)
    {
        uint node_id = forces_nodes_id(i);
        set_point_loading(q, t, node_id);
    }

    return m_fa;
}


void Forces::set_point_loading(arma::dvec q, double t, uint node_id)
{
    int dofs_per_node = m_bcs->get_dofs_per_node();

    // For spatial beam elements
    uint force_dofs_index = node_id * dofs_per_node - (dofs_per_node - 1);

    // Find indices that correspond to force_dofs_index 
    uint fa_index = arma::as_scalar(arma::find(m_la == force_dofs_index));

    // Forces 
    arma::dvec force = force_per_node(q, t, node_id);

    // Moments 
    arma::dvec moment = moment_per_node(q, t, node_id);

    // Set values to m_fa
    m_fa(arma::span(fa_index, fa_index + dofs_per_node - 1)) = 
        arma::join_vert(force, moment);
}


arma::dvec Forces::force_per_node(arma::dvec q, double t, uint node_id)
{
    arma::dvec forces;

    double fx = 0.0 * t;
    double fy = 0.0 * t;
    double fz = 0.5 * t;

    if (fx >= 0.0) { fx =  0.0; }
    if (fy >= 0.5) { fy =  0.0; }
    if (fz >= 0.5) { fz = 0.0; }

    forces = {fx, fy, fz};
    return forces;
}

arma::dvec Forces::moment_per_node(arma::dvec q, double t, uint node_id)
{
    arma::dvec moments = {0.0, 0.0};
    return moments;
}
