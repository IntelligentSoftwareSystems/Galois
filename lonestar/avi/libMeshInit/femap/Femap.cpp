#include "Femap.h"



//
// FemapInput::FemapInput(const char* fileName)
//   Constructor.  Opens input file and checks to make sure it worked.
//   Searches for "-1" indicator, identifies following record,
//   and calls appropriate function to deal with the record.
//

#ifdef GALOIS_HAS_GZ_SUPP
FemapInput::FemapInput(const char* fileName) : m_ifs() {

  std::ifstream gzfile (fileName, std::ios_base::in | std::ios_base::binary);
  m_ifs.push (boost::iostreams::gzip_decompressor ());
  m_ifs.push (gzfile);
#else
FemapInput::FemapInput(const char* fileName) : m_ifs(fileName) {
#endif


  if (m_ifs) {
    std::cout << std::endl
	 << "Femap Neutral file " << fileName << " is open for input."
	 << std::endl
	 << std::endl; 
  }
  else {
    std::cerr << "Cannot open Femap Neutral file " << fileName 
	 << ". Quitting\n";
    exit(1);
  }

  std::string s;
  int id;

  for ( m_ifs >> s; ( m_ifs >> id ) && ( s == "-1" ) ; m_ifs >> s ) {
    nextLine();
    switch ( id ) {
    case 100: _readHeader();      break;
      //case 402: _readProperty();    break;
    case 403: _readNodes();       break;
    case 404: _readElements();    break;
      //case 408: _readGroups();      break;
      //case 506: _readConstraints(); break;
    case 507: _readLoads();       break;
      //case 601: _readMaterial();    break;
    case 999:                     break;
    default:
      //std::cout << "Skipping Data Block " << id << ".  Not supported.\n";
      do { getline( m_ifs, s ); s.assign(s,0,5); } while ( s != "   -1" );
    }
  }
  std::cout << "\nDone reading Neutral File input.  Closing file " << fileName << ".\n\n";

  return;
  
}

//
// FemapInput::_readHeader()
//   Called for Data Block ID 100.  
//   Reads Neutral File Header and prints to stdout
//
void FemapInput::_readHeader()
{
  std::string s;
  m_ifs >> s;
  if (s=="<NULL>") s="";
  std::cout << "Database Title: " << s <<std::endl;

  m_ifs >> s;
  std::cout << "Created with version: " << s <<std::endl;

  m_ifs >> s;
  if (s!="-1") {
    std::cerr << "Too many records in Data Block 100.\n";
    while (s!="-1") m_ifs >> s;
  }

  return;
}

//
// FemapInput::_readProperty()
//
void FemapInput::_readProperty()
{
  std::string s, sdumm;
  std::cout << "Reading Properties.\n";
  
  femapProperty p;

  // read prop id, etc.
  getline( m_ifs, s );

  do{
    sscanf(s.c_str(), "%zd,%*d,%zd,%zd,%*d,%*d", &(p.id), &(p.matId), &(p.type));
    
    // read & print title
    m_ifs >> p.title;
    if (p.title=="<NULL>") p.title="";
    std::cout << "Id: " << p.id << " Title: " << p.title << std::endl;
    
    nextLine();
    
    // read flag[0,3]
    getline( m_ifs, s );
    sscanf(s.c_str(), "%d,%d,%d,%d", p.flag, p.flag+1, p.flag+2, p.flag+3);
    
    // skip laminate data
    int i;
    m_ifs >> i;
    nextLine();
    nextLine(i/8);
    if (i%8) nextLine();
    
    // get # of prop values & size value std::vector accordingly
    m_ifs >> p.num_val;
    p.value.resize(p.num_val);
    
    // get values
    nextLine();
    for (i=0; i<p.num_val; i++) 
      { char dumm; m_ifs >> p.value[i]; m_ifs >> dumm; }    
    
    // read num_outline
    int num_outline;
    m_ifs >> num_outline;
    nextLine();

    // skip outline point definitions
    nextLine(num_outline);
    
    _properties.push_back(p);
    _propertyIdMap[p.id] = _properties.size() - 1;
    

    // Look for more properties
    getline( m_ifs, s );
    
  }  while ( sdumm.assign(s,0,5) != "   -1" ); 
    
  return;
}

//
// FemapInput::_readNodes()      
//
void FemapInput::_readNodes()      
{
  std::string s;
  std::cout << "Reading Nodes.\n";
  femapNode n;
  getline( m_ifs, s ); 
  while ( sscanf(s.c_str(), "%zd,%*d,%*d,%*d,%*d,%d,%d,%d,%d,%d,%d,%lg,%lg,%lg,%*d,",
		 &(n.id), &(n.permBc[0]), &(n.permBc[1]), &(n.permBc[2]), &(n.permBc[3]),
		 &(n.permBc[4]), &(n.permBc[5]), &(n.x[0]), &(n.x[1]), &(n.x[2]) ) == 10 ) 
  {
    _nodes.push_back(n);
    _nodeIdMap[n.id] = _nodes.size() - 1;
    getline( m_ifs, s );
  }

  std::cout << "Read " << _nodes.size() << " nodes.\n" << std::flush;
  return;
}

//
// FemapInput::_readElements()  
//
void FemapInput::_readElements()
{
  std::string s;
  std::cout << "Reading Elements.\n";

  int nd[20];
  getline( m_ifs, s );
  int id, propId, type, topology, geomId, formulation;

  while ( sscanf(s.c_str(), "%d,%*d,%d,%d,%d,%*d,%*d,%*d,%d,%d,%*d,%*d,",
		 &(id), &(propId), &(type), &(topology), &(geomId),
		 &(formulation) ) == 6 ) 
  {
    femapElement e;
    e.id          = id;
    e.propId      = propId;
    e.type        = type;
    e.topology    = topology;
    e.geomId      = geomId;
    e.formulation = formulation;

    getline( m_ifs, s ); 
    sscanf(s.c_str(), "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,",
	   &(nd[0]),&(nd[1]),&(nd[2]),&(nd[3]),&(nd[4]),
	   &(nd[5]),&(nd[6]),&(nd[7]),&(nd[8]),&(nd[9]) );

    getline( m_ifs, s ); 
    sscanf(s.c_str(), "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,",
	   &(nd[10]),&(nd[11]),&(nd[12]),&(nd[13]),&(nd[14]),
	   &(nd[15]),&(nd[16]),&(nd[17]),&(nd[18]),&(nd[19]) );

    for (int i = 0; i < 20; i++ )
      if ( nd[i] != 0 ) e.node.push_back( nd[i] );

    nextLine(3);
    
    getline( m_ifs, s );
    int flg[4];
    sscanf(s.c_str(), "%*d,%*d,%*d,%*d,%*d,%*d,%*d,%*d,%*d,%*d,%*d,%*d,%d,%d,%d,%d,", 
	   &(flg[0]), &(flg[1]), &(flg[2]), &(flg[3]) );
    if ( flg[0]!=0 || flg[1]!=0 || flg[2]!=0 || flg[3]!=0 ) {
      std::cerr << "Unexpected node lists attatched to element " << e.id << ". Quitting.\n";
      exit(-1);
    }

    _elements.push_back(e);

    _elementIdMap[e.id] = _elements.size() - 1;

    getline( m_ifs, s );
  }
      
  std::cout << "Read " << _elements.size() << " elements.\n";

  return;
}

//
// FemapInput::_readConstraints()
//
void FemapInput::_readConstraints()
{
  std::cout << "Reading Constraints.\n";
  std::string s;
  femapConstraintSet cs;

  m_ifs >> cs.id;

  if ( static_cast<long> (cs.id) != -1 ) {
    m_ifs >> cs.title;


    const size_t NDOF = 6;
    int dofread[NDOF];

    constraint c;
    getline( m_ifs, s );
    while ( sscanf(s.c_str(), "%zd,%*d,%*d,%d,%d,%d,%d,%d,%d,%d",
          &(c.id), &(dofread[0]), &(dofread[1]), &(dofread[2]), 
          &(dofread[3]), &(dofread[4]), &(dofread[5]), &(c.ex_geom) ) == 8
        && static_cast<long> (c.id) != -1 ) {

      for (size_t i = 0; i < NDOF; ++i) {
        c.dof[i] = !(dofread[i] == 0); // 0 -> false, non-zero -> true
      }

      cs.nodalConstraint.push_back(c);
    }
    
    _constraintSets.push_back(cs);
  }
  // skip other types of constraints (geom., etc.)
  do { getline( m_ifs, s ); s.assign(s,0,5); } while ( s != "   -1" );

  return;
}

//
// FemapInput::_readLoads()      
//
void FemapInput::_readLoads()
{
  std::cout << "Reading Loads.\n";
  std::string s;
  femapLoadSet ls;

  //m_ifs >> ls.id;
  getline( m_ifs, s );
  sscanf( s.c_str(), "%zd,", &(ls.id) );  

  if ( static_cast<long> (ls.id) != -1 ) {
    getline( m_ifs, ls.title );
    std::cout << ls.title << std::endl << std::flush;
    
    getline( m_ifs, s ); 

    int tempOn_int, gravOn_int, omegaOn_int;
    sscanf(s.c_str(), "%*d,%lg,%d,%d,%d,",
	   &(ls.defTemp),&(tempOn_int),&(gravOn_int),&(omegaOn_int) );

    ls.tempOn = !(tempOn_int == 0); // 0-> false, non-zero -> true
    ls.gravOn = !(gravOn_int == 0); // 0-> false, non-zero -> true
    ls.omegaOn = !(omegaOn_int == 0); // 0-> false, non-zero -> true

    getline( m_ifs, s ); 
    sscanf(s.c_str(), "%lg,%lg,%lg,",
	   &(ls.grav[0]), &(ls.grav[1]), &(ls.grav[2]) );

    getline( m_ifs, s ); 
    sscanf(s.c_str(), "%lg,%lg,%lg,",
	   &(ls.grav[3]), &(ls.grav[4]), &(ls.grav[5]) );

    getline( m_ifs, s ); 
    sscanf(s.c_str(), "%lg,%lg,%lg,",
	   &(ls.origin[0]), &(ls.origin[1]), &(ls.origin[2]) );

    getline( m_ifs, s ); 
    sscanf(s.c_str(), "%lg,%lg,%lg,",
	   &(ls.omega[0]), &(ls.omega[1]), &(ls.omega[2]) );
    
    nextLine(14); // skip some junk we won't use 
    
    int id, type, exp;
    getline( m_ifs, s ); 
    sscanf( s.c_str(), "%d,%d,%*d,%*d,%*d,%*d,%d,", &id, &type, &exp );
    while ( id != -1 ) 
      {
	load l;
	l.id = id;
	l.type = type;
	l.is_expanded = exp;

	getline( m_ifs, s );
	sscanf(s.c_str(), "%d,%d,%d",
	       &(l.dof_face[0]), &(l.dof_face[1]), &(l.dof_face[2]) );
	getline( m_ifs, s );
	sscanf(s.c_str(), "%lg,%lg,%lg,%*g,%*g,",
	       &(l.value[0]), &(l.value[1]), &(l.value[2]) );
	nextLine(4);
	ls.loads.push_back(l);

	getline( m_ifs, s ); 
	sscanf( s.c_str(), "%d,%d,%*d,%*d,%*d,%*d,%d,", &id, &type, &exp );
      }
    
    _loadSets.push_back(ls);

  }
  
  // skip geometry-based and non-structural loads
  do { getline( m_ifs, s ); s.assign(s,0,5); } while ( s != "   -1" );

  return;
}

//
// FemapInput::_readMaterial()  
//
void FemapInput::_readMaterial()
{
  std::string s, sdumm; //temporary string
  int i;
  std::cout << "Reading Materials.\n";
  femapMaterial m;

  int functioncount;
  // read mat id, etc.
  getline( m_ifs, s );
  
  do{
    sscanf( s.c_str(), "%zd,%*d,%*d,%zd,%zd,%*d,%d", 
                      &(m.id), &(m.type), &(m.subtype), &functioncount);
    
    // read & print title
    m_ifs >> m.title;
    if (m.title=="<NULL>") m.title="";
    std::cout << "Id: " << m.id << " Title: " << m.title << std::endl;
    
    nextLine(2);
    
    // get bval[0,9]
    getline( m_ifs, s );
    sscanf(s.c_str(), "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
	   &(m.bval[0]), &(m.bval[1]), &(m.bval[2]), &(m.bval[3]), &(m.bval[4]), 
	   &(m.bval[5]), &(m.bval[6]), &(m.bval[7]), &(m.bval[8]), &(m.bval[9]));
    
    nextLine();
    
    // get ival[0,24]; 2 rows of 10 and 1 row of 5.
    for (i=0; i<2; i++) {
      getline( m_ifs, s );
      sscanf(s.c_str(), "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
	     &(m.ival[10*i+0]), &(m.ival[10*i+1]), &(m.ival[10*i+2]), 
	     &(m.ival[10*i+3]), &(m.ival[10*i+4]), &(m.ival[10*i+5]), 
	     &(m.ival[10*i+6]), &(m.ival[10*i+7]), &(m.ival[10*i+8]), &(m.ival[10*i+9]));
    }    
    getline( m_ifs, s );
    sscanf(s.c_str(), "%d,%d,%d,%d,%d",
	   &(m.ival[10*i+0]),&(m.ival[10*i+1]),
	   &(m.ival[10*i+2]),&(m.ival[10*i+3]),&(m.ival[10*i+4]));
    
    nextLine();
    
    // get mval[0,199]; 20 rows of 10.
    for (i=0; i<20; i++) {
      getline( m_ifs, s );
      sscanf(s.c_str(), "%lg,%lg,%lg,%lg,%lg,%lg,%lg,%lg,%lg,%lg",
	     &(m.mval[10*i+0]), &(m.mval[10*i+1]), &(m.mval[10*i+2]), 
	     &(m.mval[10*i+3]), &(m.mval[10*i+4]), &(m.mval[10*i+5]), 
	     &(m.mval[10*i+6]), &(m.mval[10*i+7]), &(m.mval[10*i+8]), &(m.mval[10*i+9]));
    }    
    
    // skip function data
    nextLine(14+functioncount);

    _materials.push_back(m);
    _materialIdMap[m.id] = _materials.size() - 1;

    // Look for more materials
    getline( m_ifs, s );
    
  } while ( sdumm.assign(s,0,5) != "   -1" );
  
  return;
}

  //
// FemapInput::_readGroups()     
//
void FemapInput::_readGroups () {
  std::string s;
  std::cout << "Reading Groups." << std::endl << std::flush;

  int id, need_eval;

  getline (m_ifs, s);
  sscanf (s.c_str (), "%d,%d,%*d,", &id, &need_eval);

  while (id != -1) {

    femapGroup g;

    g.id = id;
    g.need_eval = need_eval;

    getline (m_ifs, g.title);
    std::cout << g.title << std::endl << std::flush;

    getline (m_ifs, s);
    sscanf (s.c_str (), "%d,%d,%d,",
        &(g.layer[0]), &(g.layer[1]), &(g.layer_method));

    nextLine (20); // skip clipping info

    // read group rules
    size_t max;
    getline (m_ifs, s);
    sscanf (s.c_str (), "%zd,", &max);

    getline (m_ifs, s);
    groupRule r;
    sscanf (s.c_str (), "%zd,", &(r.type));
    while (static_cast<long> (r.type) != -1) {
      if (r.type < max) {
        getline (m_ifs, s);
        sscanf (s.c_str (), "%zd,%zd,%zd,%zd,",
            &(r.startID), &(r.stopID), &(r.incID), &(r.include));
        while (static_cast<long> (r.startID) != -1) {
          g.rules.push_back (r);
          getline (m_ifs, s);
          sscanf (s.c_str (), "%zd,%zd,%zd,%zd,",
              &(r.startID), &(r.stopID), &(r.incID), &(r.include));
        }
      } else
        nextLine ();

      getline (m_ifs, s);
      sscanf (s.c_str (), "%zd,", &(r.type));
    }

    // read group lists
    getline (m_ifs, s);
    sscanf (s.c_str (), "%zd,", &max);

    groupList l;
    getline (m_ifs, s);
    sscanf (s.c_str (), "%zd,", &(l.type));

    while (static_cast<long> (l.type) != -1) {
      if (l.type < max) {
        getline (m_ifs, s);
        sscanf (s.c_str (), "%d,", &id);
        while (id != -1) {
          l.entityID.push_back (id);
          getline (m_ifs, s);
          sscanf (s.c_str (), "%d,", &id);
        }
        g.lists.push_back (l);
      } else {
        nextLine ();
      }

      getline (m_ifs, s);
      sscanf (s.c_str (), "%zd,", &(l.type));
    }

    _groups.push_back (g);
    _groupIdMap[g.id] = _groups.size () - 1;

    getline (m_ifs, s);
    sscanf (s.c_str (), "%d,%d,%*d,", &id, &need_eval);
  }

  //  do { getline( m_ifs, s ); s.assign(s,0,5); } 
  //  while ( s != "   -1" );

  return;
}
// MO 1/9/01 begin
/*
inline void FemapInput::nextLine(int n=1)
{
std::string s;
  for (int i = 0; i < n; i++) getline( m_ifs, s );
  return;
}
*/
inline void FemapInput::nextLine(int n)
{
  std::string s;
  for (int i = 0; i < n; i++) {
    getline( m_ifs, s );
  }
  return;
}
inline void FemapInput::nextLine()
{
  std::string s;
  getline( m_ifs, s );
  return;
}
//  MO 1/9/01 end

size_t Femap::getNumElements(size_t t) const
{
  size_t n = 0;
  for (std::vector<femapElement>::const_iterator e = _elements.begin(); e != _elements.end(); e++ ) {
    if ( e->topology == t ) {
      n++;
    }
  }

  return n;
}

void Femap::getElements (size_t t, std::vector<femapElement>& vout) const
{
  for (std::vector<femapElement>::const_iterator e = _elements.begin(); e != _elements.end(); e++ ) {
    if ( e->topology == t ) {
      vout.push_back(*e);
    }
  }

}
