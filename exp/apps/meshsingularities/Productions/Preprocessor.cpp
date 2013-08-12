#include "Preprocessor.h"

std::vector<EquationSystem*>* Mes2DPreprocessor::preprocess(std::list<Tier * > *tier_list)
{
	int i = 0;
	std::vector<EquationSystem *> *esList = new std::vector<EquationSystem*>();

	std::list<Tier*>::iterator it = tier_list->begin();
	for (; it != tier_list->end(); ++it, ++i) {
		if (i==0) {
			EquationSystem *system = new EquationSystem((*it)->get_tier_matrix(), (*it)->get_tier_rhs(), 21);
			// in A1 we need to move 2,4,6,8 [row,col] to the top-left corner of matrix
			system->swapCols(0, 1);
			system->swapCols(2, 3);
			system->swapCols(4, 5);
			system->swapCols(6, 7);
			system->swapCols(1, 2);
			system->swapCols(3, 4);
			system->swapCols(5, 6);
			system->swapCols(2, 3);
			system->swapCols(4, 5);
			system->swapCols(3, 4);

			system->swapRows(0, 1);
			system->swapRows(2, 3);
			system->swapRows(4, 5);
			system->swapRows(6, 7);
			system->swapRows(1, 2);
			system->swapRows(3, 4);
			system->swapRows(5, 6);
			system->swapRows(2, 3);
			system->swapRows(4, 5);
			system->swapRows(3, 4);
			system->eliminate(4);

			esList->push_back(system);
		} else if (i==tier_list->size()-1) {
			EquationSystem *system = new EquationSystem(21);
			double ** tierMatrix = (*it)->get_tier_matrix();
			double *tierRhs = (*it)->get_tier_rhs();

			for (int i=0; i<17; i++) {
				for (int j=0; j<17; j++) {
					system->matrix[i+4][j+4] = tierMatrix[i][j];
				}
				system->rhs[i+4] = tierRhs[i];
			}

			for (int i=0;i<4;i++) {
				for (int j=0;j<17;j++) {
					system->matrix[i][j+4] = tierMatrix[i+17][j];
					system->matrix[j+4][i] = tierMatrix[j][i+17];
				}
			}

			for (int i=0; i<4; i++) {
				for (int j=0; j<4; j++) {
					system->matrix[i][j] = tierMatrix[i+17][j+17];
				}
				system->rhs[i] = tierRhs[i+17];
			}

			system->eliminate(4);

			esList->push_back(system);
		} else {
			EquationSystem *system = new EquationSystem((*it)->get_tier_matrix(), (*it)->get_tier_rhs(), 17);
			esList->push_back(system);
		}
	}
	return esList;
}
