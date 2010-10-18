#include "limits.h"



class Local
  {
public:
        GNode *src;
        int cur;
        bool finished;
        int minHeight;
        int minEdge;
        int relabelCur;

        void resetForRelabel()
        {
          minHeight = MAX_INT;
          minEdge = 0;
          relabelCur = 0;
        }
  };

