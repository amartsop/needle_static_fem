    // Analytical solution 
    double young_modulus = 2e11;
    double needle_length = 2.623e-1;
    double needle_radius = 6.35e-4;
    double needle_area = M_PI * pow(needle_radius, 2.0);
    double needle_density = 7850;
    double needle_mass = needle_density * needle_area * needle_length;
    double i_zz =  (M_PI / 4) * pow(needle_radius, 4.0);
   
    double md_analytical = (fy * pow(needle_length, 3.0)) / (3.0 * young_modulus * i_zz);

    std::cout << "Fem max deflection: " << md_fem << ", " << 
     "Analytical max deflection: " << md_analytical << std::endl;



    // Static solution
    arma::dvec qr = arma::zeros<arma::dvec>(3);

    arma::dvec fa = forces.get_force_fa(qr, 0);

    arma::dmat kaa = boundary_conditions.get_stiffness_matrix_kaa();

    arma::dvec qa = arma::solve<arma::dmat>(kaa, fa);
    arma::dvec qg = input_coords.get_displacement_qg();

    std::cout << qg << std::endl;
    std::cout << qa << std::endl;
    arma::dvec q = boundary_conditions.assemple_solution(qa, qg);

    
    // Post processing 
    PostProcessing post_processing(&needle);

    NeedleAnimation needle_animation(&needle, &post_processing);

    arma::dvec roa_g_g = {0, 0, 0.2};
    arma::dvec euler_angles = {0.0, 0.0, 0.0};

    while (1)
    {
        // Animate
        needle_animation.animate(roa_g_g, euler_angles, q); 
        sleep(2);
    }



// System solution 
// Known coordinates trajectory 
input_coords.update(x_current, t);
arma::dvec qg_ddot = input_coords.get_acceleration_qg_ddot();

// Forces
arma::dvec fa = forces.get_force_fa(x_current, t);

// Model 
arma::dmat zero_mat = arma::zeros<arma::dmat>(m_kaa.n_rows, m_kaa.n_rows);
arma::dmat eye_mat = arma::eye<arma::dmat>(m_kaa.n_rows, m_kaa.n_rows);
arma::dvec zero_vec = arma::zeros<arma::dvec>(m_kaa.n_rows, 1);

arma::dmat A = arma::join_vert(arma::join_horiz(zero_mat, eye_mat), 
    arma::join_horiz(- m_maa_inv * m_kaa, zero_mat));

arma::dmat b = arma::join_vert(zero_vec, 
    m_maa_inv * (fa - m_mag * qg_ddot - m_kag * qg));

arma::dmat a_exp = arma::expmat(A * t);
arma::dmat a_exp_m = arma::expmat(-A * t);
arma::dmat eye = arma::eye<arma::dmat>(A.n_rows, A.n_cols);
arma::dmat a_inv = arma::pinv(A);

arma::dvec x = a_exp * state_vector.at(0) - a_exp * a_inv * (a_exp_m - 
    eye) * b;

state_vector.push_back(x);