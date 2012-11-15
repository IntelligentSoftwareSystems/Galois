/** GMetis -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Xin Sui <xinsui@cs.utexas.edu>
 */

#ifndef ARRAYSET_H_
#define ARRAYSET_H_
#include <vector>
#include <algorithm>
#include <boost/function.hpp>

using namespace std;
template<typename T>
class ArraySet{
public:
	ArraySet(){
		setsize= 0;
	}
  ArraySet(int maxSize, boost::function<int(T)> mapToInt):indexes(maxSize){
		this->mapToInt = mapToInt;
		fill(indexes.begin(), indexes.end(), -1);
		setsize = 0;
	}
	typedef typename vector<T>::iterator iterator;
	iterator begin(){
		return setElements.begin();
	}
	iterator end(){
		return setElements.end();
	}
	bool insert(T ele){
		int index = mapToInt(ele);
		if(indexes[index]==-1){
			setsize++;
			indexes[index] = setElements.size();
			setElements.push_back(ele);
			return true;
		}
		return false;
	}
	bool erase(T ele){
		int index = mapToInt(ele);
		if(indexes[index]!=-1){
			setsize--;
			int indexEle = indexes[index];
			//swap(setElements[indexEle], setElements[setElements.size()-1]);
			setElements[indexEle] = setElements[setElements.size()-1];
			setElements.pop_back();
			indexes[mapToInt(setElements[indexEle])] = indexEle;
			indexes[index] = -1;
			return true;
		}
		return false;
	}

	void clear(){
		fill(indexes.begin(), indexes.end(), -1);
		setElements.clear();
		setsize = 0;
	}

	int size(){
//		return setElements.size();
		return setsize;
	}

private:
	vector<int> indexes;
	vector<T> setElements;
  boost::function<int(T)> mapToInt;
	int setsize;
};

#endif /* ARRAYSET_H_ */
