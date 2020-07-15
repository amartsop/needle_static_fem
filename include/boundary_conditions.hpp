#pragma once 

#include <iostream>
#include <armadillo>
#include <vector>

class BoundaryConditions
{

public:
    BoundaryConditions(int dofs_per_node, uint nodes_num, arma::ivec lg, 
        arma::dmat mass_matrix, arma::dmat stiffness_matrix);

    arma::dvec assemple_solution(arma::dvec qa, arma::dvec qg);

    /**  Getters  **/
    // Mass matrices
    arma::dmat get_mass_matrix_maa(void) { return m_maa; }
    arma::dmat get_mass_matrix_mag(void) { return m_mag; }
    arma::dmat get_mass_matrix_mga(void) { return m_mga; }
    arma::dmat get_mass_matrix_mgg(void) { return m_mgg; }

    // Stiffness matrices
    arma::dmat get_stiffness_matrix_kaa(void) { return m_kaa; }
    arma::dmat get_stiffness_matrix_kag(void) { return m_kag; }
    arma::dmat get_stiffness_matrix_kga(void) { return m_kga; }
    arma::dmat get_stiffness_matrix_kgg(void) { return m_kgg; }

    // Get id of uknown dofs
    arma::ivec get_uknown_dofs_id(void) {return m_la; }

    // Get id of known dofs 
    arma::ivec get_known_dofs_id(void) {return m_lg; }

    // Get number of nodes 
    uint get_nodes_num(void) { return m_nodes_num; }

    // Get dofs per node 
    int get_dofs_per_node(void) { return m_dofs_per_node; }

private:

    // Nodes number 
    uint m_nodes_num;

    // Dofs per node
    int m_dofs_per_node;

    // Global mass matrix 
    arma::dmat m_mass;

    // Global stiffness matrix 
    arma::dmat m_stiffness;

    // Known and unknown mass matrices
    arma::dmat m_maa, m_mag, m_mga, m_mgg;

    // Known and unknown stiffness matrices
    arma::dmat m_kaa, m_kag, m_kga, m_kgg;

    // Indices matrices 
    arma::ivec m_l, m_lg, m_la;

    // Matrix generation function
    arma::dmat matrix_generation(arma::dmat mat, arma::ivec l_index1,
        arma::ivec l_index2);
};

/** 
 *BoundaryConditions generates the mass, stiffness and force components 
 * of known and uknown quantities for the solution of the fem problem
 * @param lg Contains the indices of the known quantities 
 * @param mass_matrix Is the global mass matrix of the problem
 * @param stiffness_matrix Is the global stiffness matrix of the problem 
**/ 

BoundaryConditions::BoundaryConditions(int dofs_per_node, uint nodes_num, 
    arma::ivec lg, arma::dmat mass_matrix, arma::dmat stiffness_matrix)
{ 
    // Number of dofs per node 
    m_dofs_per_node = dofs_per_node;
    
    // Nodes number 
    m_nodes_num = nodes_num; 

    // Global mass and stiffness matrices
    m_mass = mass_matrix;
    m_stiffness = stiffness_matrix;
    m_lg = lg;

    // All coefficients
    m_l = arma::regspace<arma::imat>(1, 1, m_mass.n_rows);
    
    // Uknown coefficients
    m_la = arma::zeros<arma::imat>(m_mass.n_rows - m_lg.n_rows);
    arma::imat l_temp = m_l;
    for ( uint i = 0; i < m_lg.n_rows; i++) { l_temp(m_lg(i) - 1) = 0;}
    m_la = arma::nonzeros<arma::imat>(l_temp);
    
    // Mass matrices
    m_maa = matrix_generation(m_mass, m_la, m_la);
    m_mgg = matrix_generation(m_mass, m_lg, m_lg);
    m_mag = matrix_generation(m_mass, m_la, m_lg);
    m_mga = matrix_generation(m_mass, m_lg, m_la);

    // Stiffness matrices
    m_kaa = matrix_generation(m_stiffness, m_la, m_la);
    m_kgg = matrix_generation(m_stiffness, m_lg, m_lg);
    m_kag = matrix_generation(m_stiffness, m_la, m_lg);
    m_kga = matrix_generation(m_stiffness, m_lg, m_la);
}


arma::dmat BoundaryConditions::matrix_generation(arma::dmat mat, arma::ivec l_index1,
    arma::ivec l_index2)
{
    arma::dmat y = arma::zeros<arma::dmat>(l_index1.n_rows, l_index2.n_rows);

    for(uint i = 0; i < l_index1.n_rows; i++)
    {
        for (uint j = 0; j < l_index2.n_rows; j++)
        {
            y(i, j) = mat(l_index1(i) -1, l_index2(j) - 1);
        }
    }

    return y;
}


arma::dvec BoundaryConditions::assemple_solution(arma::dvec qa, arma::dvec qg)
{
    arma::dvec q = arma::zeros<arma::dvec>(m_mass.n_rows);
     
    for (uint i = 0; i < qa.n_rows; i++) { q(m_la(i) - 1) = qa(i); }
    for (uint i = 0; i < qg.n_rows; i++) { q(m_lg(i) - 1) = qg(i); }
    return q;
}