struct femapMaterial {
  size_t    id;
  size_t    type;
  size_t    subtype;
  std::string title;
  int    bval[10];
  int    ival[25];
  double mval[200];
};

struct femapProperty {
  size_t            id;
  size_t            matId;
  size_t            type;
  std::string         title;
  int            flag[4];
  int            num_val;
  std::vector<double> value;
};

struct femapNode {
  size_t    id;
  double x[3];
  int permBc[6];
};

struct femapElement {
  size_t id;
  size_t propId;
  size_t type;
  size_t topology;
  size_t geomId;
  int formulation;  // int is a guess--documentation doesn't give type
  std::vector<size_t> node;
};

struct constraint {
  size_t  id;
  bool dof[6];
  int  ex_geom;
};

struct femapConstraintSet {
  size_t                id;
  std::string             title;
  std::vector<constraint> nodalConstraint;
};

struct load {
  size_t    id;
  size_t    type;
  int    dof_face[3];
  double value[3];
  bool   is_expanded;
};

struct femapLoadSet {
  size_t          id;
  std::string       title;
  double       defTemp;
  bool         tempOn;
  bool         gravOn;
  bool         omegaOn;
  double       grav[6];
  double       origin[3];
  double       omega[3];
  std::vector<load> loads;
};

struct groupRule {
  size_t type;
  size_t startID;
  size_t stopID; 
  size_t incID;
  size_t include;
};

struct groupList {
  size_t type;
  std::vector<size_t> entityID;
};

struct femapGroup {
  size_t id;
  short int need_eval;
  std::string title;
  int layer[2];
  int layer_method;
  std::vector<groupRule> rules;
  std::vector<groupList> lists;
};
