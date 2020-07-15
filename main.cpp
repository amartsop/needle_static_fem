#include <iostream>
#include <armadillo>
#include <vector>
#include <chrono>
#include "gnuplot-iostream.h"

#include "spatial_beam.hpp"
#include "boundary_conditions.hpp"
#include "input_coordinates.hpp"
#include "forces.hpp"
#include "linear_fem_model.hpp"

#include "numerical_integration.hpp"
#include "post_processing.hpp"
#include "needle_animation.hpp"

int main(int argc, char *argv[])
{
    uint elements_num = 3;
    uint nodes_num = elements_num + 1;
    uint dofs_per_node = 5;
    uint dofs_num = dofs_per_node * elements_num + dofs_per_node;

    // Needle object
    SpatialBeam needle(elements_num, dofs_num); 

    // Global Matrices
    arma::dmat mff = needle.get_mass_matrix();
    arma::dmat kff = needle.get_stiffness_matrix();

    // Boundary Conditions
    arma::ivec lg = {1, 2, 3, 4, 5};
    BoundaryConditions boundary_conditions(dofs_per_node, nodes_num, lg, mff, kff);

    arma::dmat maa = boundary_conditions.get_mass_matrix_maa();
    arma::dmat kaa = boundary_conditions.get_stiffness_matrix_kaa();

    // Input coordinates 
    InputCoordinates input_coords(&boundary_conditions);

    //Forces 
    Forces forces(&boundary_conditions);

    // IDs of uknown nodes 
    arma::ivec la = boundary_conditions.get_uknown_dofs_id();
    
    // /********************* Simulation ************************/ 
    // Initial conditions
    arma::dvec qa0 = arma::zeros<arma::dvec>(la.n_rows);
    arma::dvec qa0_dot = arma::zeros<arma::dvec>(la.n_rows);

    // Problem model 
    LinearFemModel model(&boundary_conditions, &input_coords, &forces);
    
    // State vector initialization
    std::vector<arma::dvec> state_vector;
    arma::dvec state0 = arma::join_vert(qa0, qa0_dot);
    state_vector.push_back(state0);

    // Reaction forces 
    std::vector<double> fx, fy, fz;
    std::vector<double> my, mz;

    // Post processing 
    PostProcessing post_processing(&needle);

    // Timing
    double t_final = 10.0; // Final time (s)
    double fs = 5e4;  // Simulation frequency (Hz)
    double h = 1.0 / fs; // Integration time step (s)
    double t = 0; // Initial time (s) 

    // Time vector initialization 
    std::vector<double> time_vector;
    time_vector.push_back(t);

    // Problem solver 
    NumericalIntegration ni(&model, h, 4);

    // Iteration counter 
    uint counter = 0;

    // Deformation vector 
    std::vector<arma::dvec> r_vec;

    
    while (t <= t_final)
    {
        // System solution 
        arma::dvec x = ni.solve(state_vector.at(counter), t);
        state_vector.push_back(x);

        // Reaction forces 
        arma::dmat reaction_forces = model.get_reaction_forces();

        fx.push_back(reaction_forces(0)); fy.push_back(reaction_forces(1));
        fz.push_back(reaction_forces(2)); my.push_back(reaction_forces(3));
        mz.push_back(reaction_forces(4));

        if (!x.is_finite()) {
            std::cout << "Error" << std::endl; 
            break; 
        };

        // Update time vector 
        time_vector.push_back(t);

        // Update time and counter
        t += h; counter += 1;

        std::cout << t << std::endl;
    }

    // Plot reaction forces 
    Gnuplot gp;
    arma::dvec t_vec(time_vector);
    arma::dvec fx_vec(fx), fy_vec(fy), fz_vec(fz);
    arma::dvec my_vec(my), mz_vec(mz);

    arma::dmat t_fx = arma::join_horiz(t_vec.rows(0, t_vec.n_rows - 2), fx_vec);
    arma::dmat t_fy = arma::join_horiz(t_vec.rows(0, t_vec.n_rows - 2), fy_vec);
    arma::dmat t_fz = arma::join_horiz(t_vec.rows(0, t_vec.n_rows - 2), fz_vec);
    arma::dmat t_my = arma::join_horiz(t_vec.rows(0, t_vec.n_rows - 2), my_vec);
    arma::dmat t_mz = arma::join_horiz(t_vec.rows(0, t_vec.n_rows - 2), mz_vec);

    gp << "plot '-' with lines \n";
    gp.send1d(t_fz);

    // Animation
    NeedleAnimation needle_animation(&needle, &post_processing);
    arma::dvec roa_g_g = {0.0, 0.0, 0.0};
    arma::dvec euler_angles = {0.0, 0.0, 0.0};
    double animation_fs = 1e2; // Animation frequency
    int steps = fs / animation_fs;

    for(size_t i = 0; i <= state_vector.size(); i = i + steps)
    {
        arma::dvec x_current = state_vector.at(i);
        arma::dvec qa = x_current.rows(0, la.n_rows - 1);
        arma::dvec qg = input_coords.get_displacement_qg();
        arma::dvec q = boundary_conditions.assemple_solution(qa, qg);
        needle_animation.animate(roa_g_g, euler_angles, q);
    }


}


    // // Animation 
    // NeedleAnimation needle_animation(&needle, &post_processing);
    // auto start = std::chrono::steady_clock::now();
    // double f_animation = 1000.0; 

    // // Rigid body centroid position
    // arma::dvec roa_g_g = {0, 0, 0.2};
    // arma::dvec euler_angles = {0.0, 0.0, 0.0};


    // // Simulation time 
    // double t = 0;

    // // Counter 
    // uint i = 0;

    // while(1)
    // {
    //     auto end = std::chrono::steady_clock::now();
    //     std::chrono::duration<double> elapsed_seconds = end - start;
        
    //     if (elapsed_seconds.count() >= 1 / f_animation)
    //     {
    //         // Solution  
    //         arma::dvec x_current = state_vector.at(i);
    //         arma::dvec qa = x_current.rows(0, la.n_rows / 2 - 1);
    //         arma::dvec qg = input_coords.get_displacement_qg();
    //         arma::dvec q = boundary_conditions.assemple_solution(qa, qg);
            
    //         needle_animation.animate(roa_g_g, euler_angles, q); 
    //     }
         
    //     // System solution 
    //     arma::dvec x = ni.solve(state_vector.at(i), t);
    //     state_vector.push_back(x);
    //     t += h; i += 1;

    //     // Update time 
    //     start = end;

    //     std::cout << t << std::endl;
    // }
    
    
