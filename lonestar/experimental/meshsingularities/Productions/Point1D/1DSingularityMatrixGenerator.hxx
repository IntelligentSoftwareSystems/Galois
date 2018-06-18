#ifndef __1DSINGULARITYMATRIXGENERATOR_H_INCLUDED__
#define __1DSINGULARITYMATRIXGENERATOR_H_INCLUDED__

void get_matrix_and_rhs(int matrix_size, double*** matrix_p, double** rhs_p,
                        double l_boundary_condition,
                        double r_boundary_condition);

#endif
