

struct partInfo {
  unsigned partNum;
  unsigned partMask;
  unsigned partWeight;
  unsigned totalWeight;
  unsigned partSize;
  int neededParts;
  GGraph* graph;
  int tryPart;
  
  partInfo(GGraph* g,unsigned size, unsigned tw, int tryN, int neededParts=1, unsigned mw = 0)
  :partNum(0), partMask(1), partWeight(mw), totalWeight(tw), graph(g), neededParts(neededParts), partSize(size), tryPart(tryN) {
}

  partInfo() { } 

  partInfo split() {
    partInfo np(*this);
    np.partNum = partNum | partMask;
    partMask <<= 1;
    np.partMask = partMask;
    np.neededParts = neededParts/2; 
    neededParts -= np.neededParts;
    np.partWeight = 0;
    np.tryPart = tryPart;
    return np;
  }

/*  void splitre() { 
    partMask <<= 1;
    neededParts -= neededParts/2;
  }*/


};

std::ostream& operator<<(std::ostream& os, const partInfo& p);
partInfo bisect_GGP(partInfo& oldPart);

struct parallelBisect {
	//bisect_policy bisect;
	partInfo** parts;
	parallelBisect(partInfo** parts): parts(parts) {
	}

	void operator()(partInfo &item, Galois::UserContext<partInfo> &workList) {
		partInfo newPart = bisect_GGP(item);

		if(item.neededParts >1)
			workList.push(item);
		else	
			parts[item.tryPart][item.partNum] = item;
		if(newPart.neededParts >1)
			workList.push(newPart);	
		else
			parts[item.tryPart][newPart.partNum] = newPart;
	}
};


