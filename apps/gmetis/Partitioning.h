

struct partInfo {
  unsigned partNum;
  unsigned partMask;
  unsigned partWeight;
  unsigned totalWeight;
  GGraph* graph;
  
  partInfo(GGraph* g, unsigned tw, unsigned mw = 0)
  :partNum(0), partMask(1), partWeight(mw), totalWeight(tw), graph(g) {}

  partInfo() { } 

  partInfo split() {
    partInfo np(*this);
    np.partNum = partNum | partMask;
    partMask <<= 1;
    np.partMask = partMask;
    np.partWeight = 0;
    return np;
  }
};

std::ostream& operator<<(std::ostream& os, const partInfo& p);
partInfo bisect_GGP(partInfo& oldPart);
