/*
 * DataManager.h
 *
 *  Created on: Jan 25, 2011
 *      Author: xinsui
 */

#ifndef DATAMANAGER_H_
#define DATAMANAGER_H_

#include "Tuple.h"
#include "Element.h"

#include <string>
#include <fstream>
#include <istream>
#include <algorithm>
#include <climits>

using namespace std;



class DataManager{
public :
	static DTTuple t1;
	static DTTuple t2;
	static DTTuple t3;
	static double MIN_X, MIN_Y, MAX_X, MAX_Y;
		
	static void readTuplesFromFile(string filename,
			std::vector<DTTuple>* tuples){
		

		std::ifstream scanner(filename.c_str());
		double x, y;
		int k;
		scanner>>k;
		tuples->resize(k);
		for(int i=0;i<k;++i){
			scanner>>x>>y;
			if (x < MIN_X)
				MIN_X = x;
			else if (x > MAX_X)
				MAX_X = x;
			if (y < MIN_Y)
				MIN_Y = y;
			else if (y > MAX_Y)
				MAX_Y = y;
			(*tuples)[i]=DTTuple(x, y);					
		}

		double width = MAX_X - MIN_X;
		double height = MAX_Y - MIN_Y;
		double centerX = MIN_X + width / 2;
		double centerY = MIN_Y + height / 2;
		double maxNum = max(width, height);

		DataManager::t1.setX(centerX);
		DataManager::t1.setY(centerY + 3*maxNum);
		DataManager::t2.setX(centerX - 3*maxNum);
		DataManager::t2.setY(centerY - 2*maxNum);
		DataManager::t3.setX(centerX + 3*maxNum);
		DataManager::t3.setY(centerY - 2*maxNum);
	}
};


#endif /* DATAMANAGER_H_ */
