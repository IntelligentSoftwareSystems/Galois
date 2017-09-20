/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <iostream>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <dai/bp.h>
#include <dai/util.h>
#include <dai/properties.h>
#include "Galois/Galois.h"
#include "llvm/Support/CommandLine.h"
#include "Galois/PriorityScheduling.h"
#include "Galois/Timer.h"

static galois::runtime::PerThreadStorage<int> counters;
extern long GlobalTime;

namespace dai {

template<class T, class Compare=std::less<T>, class Alloc=std::allocator<T> >
class PairingHeap: private boost::noncopyable {
  /*
   * Conceptually a pairing heap is an n-ary tree. We represent this using
   * a fixed number of fields:
   *  left: the node's first child
   *  prev, next: a doubly linked list of a node's siblings, where the first
   *    child's prev points to the parent, i.e., x->left->prev == x.
   */
  struct Node {
    T value;
    Node* left;
    Node* next;
    Node* prev;
    Node(T v) : value(v), left(NULL), next(NULL), prev(NULL) { }
  };

public:
  typedef typename Alloc::template rebind<Node>::other allocator_type;

private:
  Compare comp;
  const allocator_type& alloc;
  std::vector<Node*, typename allocator_type::template rebind<Node*>::other> m_tree;
  Node* m_root;
  int m_capacity;

  allocator_type get_alloc() {
    return allocator_type(alloc);
  }

  /**
   * Merge two trees together, preserving heap property. Make the least tree the
   * parent of the other. Return new root.
   */
  Node* merge(Node* a, Node* b) {
    DAI_ASSERT(!a->next);
    DAI_ASSERT(!b->next);
    DAI_ASSERT(!a->prev);
    DAI_ASSERT(!b->prev);

    Node *first = a, *second = b;
    if (comp(b->value, a->value)) {
      first = b;
      second = a;
    }

    second->prev = first;
    second->next = first->left;
    if (first->left)
      first->left->prev = second;
    first->left = second;

    DAI_ASSERT(!first->prev);
    DAI_ASSERT(!first->next);
    return first;
  }

  /**
   * Heapify children, return new root.
   */
  Node* mergeChildren(Node* root) {
    if (!root->left)
      return NULL;

    // Breakup kids
    for (Node* kid = root->left; kid; kid = kid->next) {
      m_tree.push_back(kid);
      kid->prev->next = NULL;
      kid->prev = NULL;
    }

    Node* retval;
    if (m_tree.size() == 1) {
      retval = m_tree[0];
    } else {
      // Merge in pairs
      unsigned i = 0;
      for (; i + 1 < m_tree.size(); i += 2) {
        m_tree[i] = merge(m_tree[i], m_tree[i+1]);
      }

      // Merge pairs together
      if (i != m_tree.size()) {
        // odd number of siblings and m_tree[i] is the last
        m_tree[i-2] = merge(m_tree[i-2], m_tree[i]);
      }
      for (unsigned j = (m_tree.size() >> 1) - 1; j > 0; --j) {
        m_tree[2*(j-1)] = merge(m_tree[2*(j-1)], m_tree[2*j]);
      }

      retval = m_tree[0];
    }
    m_tree.clear();
    return retval;
  }

  /**
   * Remove and deallocate root node and return new root.
   */
  Node* deleteMin(Node* root) {
    Node* retval = mergeChildren(root);
    get_alloc().destroy(root);
    get_alloc().deallocate(root, 1);
    return retval;
  }

  /**
   * Remove node from heap, putting node in it's own heap.
   */
  void detach(Node* node) {
    if (node == m_root)
      return;

    if (node->prev->left == node) {
      // node is someone's left-most child
      node->prev->left = node->next;
    } else {
      node->prev->next = node->next;
    }
    if (node->next)
      node->next->prev = node->prev;
    node->next = NULL;
    node->prev = NULL;
  }

public:
  typedef Node* Handle;

  PairingHeap(const allocator_type& a = allocator_type()):
    alloc(a), m_tree(a), m_root(NULL) { }

  ~PairingHeap() {
    while (!empty()) {
      pollMin();
    }
  }

  bool empty() const {
    return m_root == NULL;
  }

  Handle add(T x) {
    typename allocator_type::pointer node = get_alloc().allocate(1);
    get_alloc().construct(node, Node(x));

    if (!m_root)
      m_root = node;
    else
      m_root = merge(m_root, node);

    return node;
  }

  T value(Handle node) {
    return node->value;
  }

  void decreaseKey(Handle node, const T& val) {
    DAI_ASSERT(comp(val, node->value));

    node->value = val;
    if (node != m_root) {
      detach(node);
      m_root = merge(m_root, node);
    }
  }

  void deleteNode(Handle& node) {
    if (node == m_root) {
      pollMin();
    } else {
      detach(node);
      node = deleteMin(node);
      if (node)
        m_root = merge(m_root, node);
    }
    node = NULL;
  }

  std::pair<bool,T> pollMin() {
    if (empty())
      return std::make_pair(false, T());
    T retval = m_root->value;
    m_root = deleteMin(m_root);

    return std::make_pair(true, retval);
  }
};

using namespace std;


#define DAI_BP_FAST 1
/// \todo Make DAI_BP_FAST a compile-time choice, as it is a memory/speed tradeoff


void BP::setProperties( const PropertySet &opts ) {
    DAI_ASSERT( opts.hasKey("tol") );
    DAI_ASSERT( opts.hasKey("logdomain") );
    DAI_ASSERT( opts.hasKey("updates") );

    props.tol = opts.getStringAs<Real>("tol");
    props.logdomain = opts.getStringAs<bool>("logdomain");
    props.updates = opts.getStringAs<Properties::UpdateType>("updates");

    if( opts.hasKey("maxiter") )
        props.maxiter = opts.getStringAs<size_t>("maxiter");
    else
        props.maxiter = 10000;
    if( opts.hasKey("maxtime") )
        props.maxtime = opts.getStringAs<Real>("maxtime");
    else
        props.maxtime = INFINITY;
    if( opts.hasKey("verbose") )
        props.verbose = opts.getStringAs<size_t>("verbose");
    else
        props.verbose = 0;
    if( opts.hasKey("damping") )
        props.damping = opts.getStringAs<Real>("damping");
    else
        props.damping = 0.0;
    if( opts.hasKey("inference") )
        props.inference = opts.getStringAs<Properties::InfType>("inference");
    else
        props.inference = Properties::InfType::SUMPROD;

    if (opts.hasKey("worklist"))
        props.worklist = opts.getStringAs<std::string>("worklist");
    else
        props.worklist = "";
}


PropertySet BP::getProperties() const {
    PropertySet opts;
    opts.set( "tol", props.tol );
    opts.set( "maxiter", props.maxiter );
    opts.set( "maxtime", props.maxtime );
    opts.set( "verbose", props.verbose );
    opts.set( "logdomain", props.logdomain );
    opts.set( "updates", props.updates );
    opts.set( "damping", props.damping );
    opts.set( "inference", props.inference );
    return opts;
}


string BP::printProperties() const {
    stringstream s( stringstream::out );
    s << "[";
    s << "tol=" << props.tol << ",";
    s << "maxiter=" << props.maxiter << ",";
    s << "maxtime=" << props.maxtime << ",";
    s << "verbose=" << props.verbose << ",";
    s << "logdomain=" << props.logdomain << ",";
    s << "updates=" << props.updates << ",";
    s << "damping=" << props.damping << ",";
    s << "inference=" << props.inference << "]";
    return s.str();
}


void BP::construct() {
    // create edge properties
    _edges.clear();
    _edges.reserve( nrVars() );
    _edge2lut.clear();
    if( props.updates == Properties::UpdateType::SEQMAX )
        _edge2lut.reserve( nrVars() );
    for( size_t i = 0; i < nrVars(); ++i ) {
        _edges.push_back( vector<EdgeProp>() );
        _edges[i].reserve( nbV(i).size() );
        if( props.updates == Properties::UpdateType::SEQMAX ) {
            _edge2lut.push_back( vector<LutType::iterator>() );
            _edge2lut[i].reserve( nbV(i).size() );
        }
        diaforeach( const Neighbor &I, nbV(i) ) {
            EdgeProp newEP;
            newEP.message = Prob( var(i).states() );
            newEP.newMessage = Prob( var(i).states() );

            if( DAI_BP_FAST ) {
                newEP.index.reserve( factor(I).nrStates() );
                for( IndexFor k( var(i), factor(I).vars() ); k.valid(); ++k )
                    newEP.index.push_back( k );
            }

            newEP.residual = 0.0;
            _edges[i].push_back( newEP );
            if( props.updates == Properties::UpdateType::SEQMAX )
                _edge2lut[i].push_back( _lut.insert( make_pair( newEP.residual, make_pair( i, _edges[i].size() - 1 ))) );
        }
    }

    // create old beliefs
    _oldBeliefsV.clear();
    _oldBeliefsV.reserve( nrVars() );
    for( size_t i = 0; i < nrVars(); ++i )
        _oldBeliefsV.push_back( Factor( var(i) ) );
    _oldBeliefsF.clear();
    _oldBeliefsF.reserve( nrFactors() );
    for( size_t I = 0; I < nrFactors(); ++I )
        _oldBeliefsF.push_back( Factor( factor(I).vars() ) );
    
    // create update sequence
    _updateSeq.clear();
    _updateSeq.reserve( nrEdges() );
    for( size_t I = 0; I < nrFactors(); I++ )
        diaforeach( const Neighbor &i, nbF(I) )
            _updateSeq.push_back( Edge( i, i.dual ) );
}


void BP::init() {
    Real c = props.logdomain ? 0.0 : 1.0;
    for( size_t i = 0; i < nrVars(); ++i ) {
        diaforeach( const Neighbor &I, nbV(i) ) {
            message( i, I.iter ).fill( c );
            newMessage( i, I.iter ).fill( c );
            if( props.updates == Properties::UpdateType::SEQMAX ) {
                updateResidual( i, I.iter, 0.0 );
            }
        }
    }
    _iters = 0;
}


void BP::findMaxResidual( size_t &i, size_t &_I ) {
    DAI_ASSERT( !_lut.empty() );
    LutType::const_iterator largestEl = _lut.end();
    --largestEl;
    i  = largestEl->second.first;
    _I = largestEl->second.second;
}


Prob BP::calcIncomingMessageProduct( size_t I, bool without_i, size_t i ) const {
    Factor Fprod( factor(I) );
    Prob &prod = Fprod.p();
    if( props.logdomain )
        prod.takeLog();

    // Calculate product of incoming messages and factor I
    diaforeach( const Neighbor &j, nbF(I) )
        if( !(without_i && (j == i)) ) {
            // prod_j will be the product of messages coming into j
            Prob prod_j( var(j).states(), props.logdomain ? 0.0 : 1.0 );
            diaforeach( const Neighbor &J, nbV(j) )
                if( J != I ) { // for all J in nb(j) \ I
                    if( props.logdomain )
                        prod_j += message( j, J.iter );
                    else
                        prod_j *= message( j, J.iter );
                }

            // multiply prod with prod_j
            if( !DAI_BP_FAST ) {
                // UNOPTIMIZED (SIMPLE TO READ, BUT SLOW) VERSION
                if( props.logdomain )
                    Fprod += Factor( var(j), prod_j );
                else
                    Fprod *= Factor( var(j), prod_j );
            } else {
                // OPTIMIZED VERSION
                size_t _I = j.dual;
                // ind is the precalculated IndexFor(j,I) i.e. to x_I == k corresponds x_j == ind[k]
                const ind_t &ind = index(j, _I);

                for( size_t r = 0; r < prod.size(); ++r ) {
                    if( props.logdomain )
                        prod.set( r, prod[r] + prod_j[ind[r]] );
                    else
                        prod.set( r, prod[r] * prod_j[ind[r]] );
                }
            }
    }
    return prod;
}


void BP::calcNewMessage( size_t i, size_t _I ) {
    // calculate updated message I->i
    size_t I = nbV(i,_I);

    Prob marg;
    if( factor(I).vars().size() == 1 ) // optimization
        marg = factor(I).p();
    else {
        Factor Fprod( factor(I) );
        Prob &prod = Fprod.p();
        prod = calcIncomingMessageProduct( I, true, i );

        if( props.logdomain ) {
            prod -= prod.max();
            prod.takeExp();
        }

        // Marginalize onto i
        if( !DAI_BP_FAST ) {
            // UNOPTIMIZED (SIMPLE TO READ, BUT SLOW) VERSION
            if( props.inference == Properties::InfType::SUMPROD )
                marg = Fprod.marginal( var(i) ).p();
            else
                marg = Fprod.maxMarginal( var(i) ).p();
        } else {
            // OPTIMIZED VERSION 
            marg = Prob( var(i).states(), 0.0 );
            // ind is the precalculated IndexFor(i,I) i.e. to x_I == k corresponds x_i == ind[k]
            const ind_t ind = index(i,_I);
            if( props.inference == Properties::InfType::SUMPROD ) 
                for( size_t r = 0; r < prod.size(); ++r ) {
                    marg.set( ind[r], marg[ind[r]] + prod[r] );
                }
            else
                for( size_t r = 0; r < prod.size(); ++r )
                    if( prod[r] > marg[ind[r]] )
                        marg.set( ind[r], prod[r] );
            marg.normalize();
        }
    }

    // Store result
    if( props.logdomain )
        newMessage(i,_I) = marg.log();
    else
        newMessage(i,_I) = marg;

    // Update the residual if necessary
    if( props.updates == Properties::UpdateType::SEQMAX )
        updateResidual( i, _I , dist( newMessage( i, _I ), message( i, _I ), DISTLINF ) );
}

void BP::runProcess(const Task& t, std::vector<std::vector<EdgeData> >& edgeData,
    unsigned& count, galois::UserContext<Task>& ctx) {
      size_t i = t.i;
      size_t _I = t._I;

      edgeData[i][_I].lock.get(galois::MethodFlag::WRITE);
      // Acquire neighborhood
      //diaforeach(const Neighbor &J, nbV(i)) {
      //  diaforeach(const Neighbor &j, nbF(J)) {
      //    edgeData[j][j.dual].lock.getData(galois::MethodFlag::ALL);
      //  }
      //}

      if (edgeData[i][_I].round != t.round) {
        return;
      }

      // update the message with the largest residual
      if ((*counters.getLocal())++ == 1000) {
        *counters.getLocal() = 0;
        //Indexer I;
        //std::cout << "AAAA: " << count << " " << t.dist << " " << t.round << "\n";
        if (__sync_fetch_and_add(&count, 1000) >= _updateSeq.size()) {
            ctx.breakLoop();
            return;
        }
      }

      updateMessage( i, _I );

      Real d = -dist(newMessage(i,_I), message(i, _I), DISTLINF);
      
      ctx.push(Task(i, _I, t.round + 1, d));
      edgeData[i][_I].round = t.round + 1;

      // I->i has been updated, which means that residuals for all
      // J->j with J in nb[i]\I and j in nb[J]\i have to be updated
      diaforeach( const Neighbor &J, nbV(i) ) {
          if( J.iter != _I ) {
              diaforeach( const Neighbor &j, nbF(J) ) {
                  size_t _J = j.dual;
                  if( j != i ) {
                      calcNewMessage( j, _J );
                      Real d = -dist(newMessage(j,_J), message(j, _J), DISTLINF);
                      ctx.push(Task(j, _J, t.round + 1, d));
                      edgeData[j][_J].round = t.round + 1;
                  }
              }
          }
      }
}

void BP::runInitialize(size_t index, std::vector<Task>& initial, 
    std::vector<std::vector<EdgeData> >& edgeData) {
  Task& t = initial[index];
  Real d = -dist(newMessage(t.i,t._I), message(t.i, t._I), DISTLINF);
  t.dist = d;
  edgeData[t.i][t._I].round = 0;
}

void BP::runComputeDiffs(size_t v, galois::GReduceMax<Real>& acc) {
  if (v < nrVars()) {
    size_t i = v;
    Factor b(beliefV(i));
    acc.update(dist(b, _oldBeliefsV[i], DISTLINF));
    _oldBeliefsV[i] = b;
  } else {
    size_t I = v - nrVars();
    Factor b( beliefF(I) );
    acc.update(dist(b, _oldBeliefsF[I], DISTLINF));
    _oldBeliefsF[I] = b;
  }
}

// BP::run does not check for NANs for performance reasons
// Somehow NaNs do not often occur in BP...
Real BP::run() {
    if( props.verbose >= 1 )
        cerr << "Starting " << identify() << "...";
    if( props.verbose >= 3)
        cerr << endl;

    double tic = toc();

    // do several passes over the network until maximum number of iterations has
    // been reached or until the maximum belief difference is smaller than tolerance
    Real maxDiff = INFINITY;
    std::vector<size_t> integers;
    for (size_t i = 0; i < nrVars() + nrFactors(); ++i)
      integers.push_back(i);
    for( ; _iters < props.maxiter && maxDiff > props.tol && (toc() - tic) < props.maxtime; _iters++ ) {
        if( props.updates == Properties::UpdateType::SEQMAX ) {
            if( _iters == 0 ) {
                // do the first pass
                for( size_t i = 0; i < nrVars(); ++i )
                  diaforeach( const Neighbor &I, nbV(i) )
                      calcNewMessage( i, I.iter );

                // XXX: Initialize everything to known state
                _lut.clear();
                for( size_t i = 0; i < nrVars(); ++i ) {
                    diaforeach( const Neighbor &I, nbV(i) ) {
                        size_t _I = I.iter;
                        EdgeProp* pEdge = &_edges[i][_I];
                        Real r = i*nrVars() + _I;
                        pEdge->residual = r;
                        _edge2lut[i][_I] = _lut.insert(make_pair(r, make_pair(i, _I)));
                    }
                }
            }
            // Maximum-Residual BP [\ref EMK06]
            for( size_t t = 0; t < _updateSeq.size(); ++t ) {
                // update the message with the largest residual
                size_t i, _I;
                findMaxResidual( i, _I );
                updateMessage( i, _I );

                // I->i has been updated, which means that residuals for all
                // J->j with J in nb[i]\I and j in nb[J]\i have to be updated
                diaforeach( const Neighbor &J, nbV(i) ) {
                    if( J.iter != _I ) {
                        diaforeach( const Neighbor &j, nbF(J) ) {
                            size_t _J = j.dual;
                            if( j != i )
                                calcNewMessage( j, _J );
                        }
                    }
                }
            }
        } else if( props.updates == Properties::UpdateType::PARALL ) {
            // Parallel updates
            for( size_t i = 0; i < nrVars(); ++i )
                diaforeach( const Neighbor &I, nbV(i) )
                    calcNewMessage( i, I.iter );

            for( size_t i = 0; i < nrVars(); ++i )
                diaforeach( const Neighbor &I, nbV(i) )
                    updateMessage( i, I.iter );
        } else if ( props.updates == Properties::UpdateType::SEQFIX 
            || props.updates == Properties::UpdateType::SEQRND ) {
            // Sequential updates
            if( props.updates == Properties::UpdateType::SEQRND )
                random_shuffle( _updateSeq.begin(), _updateSeq.end(), rnd );

            diaforeach( const Edge &e, _updateSeq ) {
                calcNewMessage( e.first, e.second );
                updateMessage( e.first, e.second );
            }
        } else if ( props.updates == Properties::UpdateType::SEQPRI ) {
          PairingHeap<Task> heap;
          std::vector<std::vector<PairingHeap<Task>::Handle > > edge2handle;

          for( size_t i = 0; i < nrVars(); ++i ) {
              edge2handle.push_back(std::vector<PairingHeap<Task>::Handle >(nbV(i).size()));     
              diaforeach( const Neighbor &I, nbV(i) ) {
                  if (_iters == 0) {
                      calcNewMessage( i, I.iter );
                      Real d = -(i*nrVars() + I.iter);
                      Task t(i, I.iter, 0, d);
                      edge2handle[i][I.iter] = heap.add(t);
                  } else {
                      Real d = -dist(newMessage(i,I.iter), message(i, I.iter), DISTLINF);
                      Task t(i, I.iter, 0, d);
                      edge2handle[i][I.iter] = heap.add(t);
                  }
              }
          }

          std::pair<bool,Task> task = heap.pollMin();
          unsigned iters = 0;
          while (task.first) {
              // update the message with the largest residual
              size_t i = task.second.i;
              size_t _I = task.second._I;
              if (iters++ >= _updateSeq.size()) {
                  edge2handle[i][_I] = heap.add(task.second);
                  break;
              }

              updateMessage( i, _I );
              Real d = -dist(newMessage(i,_I), message(i, _I), DISTLINF);
//              d = 0.0;
              edge2handle[i][_I] = heap.add(Task(i, _I, task.second.round, d));

              // I->i has been updated, which means that residuals for all
              // J->j with J in nb[i]\I and j in nb[J]\i have to be updated
              diaforeach( const Neighbor &J, nbV(i) ) {
                  if( J.iter != _I ) {
                      diaforeach( const Neighbor &j, nbF(J) ) {
                          size_t _J = j.dual;
                          if( j != i ) {
                              calcNewMessage( j, _J );
                              Real d = -dist(newMessage(j,_J), message(j, _J), DISTLINF);
                              heap.deleteNode(edge2handle[j][_J]);
                              edge2handle[j][_J] = heap.add(Task(j, _J, task.second.round, d));
                          }
                      }
                  }
              }
              task = heap.pollMin();
          }
        } else if ( props.updates == Properties::UpdateType::SEQPRIASYNC ) {
          // Asynchronous with priority queue
          static std::vector<Task> initial;
          static std::vector<size_t> initialIndex;
          static std::vector<std::vector<EdgeData> > edgeData;
          
          if (_iters == 0) {
              for( size_t i = 0; i < nrVars(); ++i ) {
                  edgeData.push_back(std::vector<EdgeData>(nbV(i).size()));
                  diaforeach( const Neighbor &I, nbV(i) ) {
                          calcNewMessage( i, I.iter );
                          Real d = -(i*nrVars() + I.iter);
                          Task t(i, I.iter, 0, d);
                          initial.push_back(t);
                          initialIndex.push_back(initialIndex.size());
                          edgeData[i][I.iter] = EdgeData();
                  }
              }
          } else {
              galois::for_each(initialIndex.begin(), initialIndex.end(), 
                  Initialize(*this, initial, edgeData));
              std::sort(initial.begin(), initial.end(), TaskGreater());

              //for( size_t i = 0; i < nrVars(); ++i ) {
              //    diaforeach( const Neighbor &I, nbV(i) ) {
              //          Real d = -dist(newMessage(i,I.iter), message(i, I.iter), DISTLINF);
              //          Task t(i, I.iter, 0, d);
              //          initial.push_back(t);
              //          edgeData[i][I.iter].round = 0;
              //    }
              //}
          }

          using namespace galois::WorkList;
          typedef dChunkedFIFO<64> dChunk;
          typedef ChunkedFIFO<64> Chunk;
          typedef OrderedByIntegerMetric<Indexer,dChunk> OBIM;
          unsigned count = 0;

          galois::Timer t;
          t.start();
          Exp::PriAuto<64, Indexer, OBIM, TaskLess,TaskGreater>::for_each(initial.begin(), initial.end(), Process(*this, edgeData, count));
          t.stop();
          GlobalTime += t.get();
        } else {
          abort();
        }

        galois::GReduceMax<Real> accMaxDiff;
        galois::for_each(integers.begin(), integers.end(), ComputeDiffs(*this, accMaxDiff));

        // calculate new beliefs and compare with old ones
        //maxDiff = -INFINITY;
        //for( size_t i = 0; i < nrVars(); ++i ) {
        //    Factor b( beliefV(i) );
        //    maxDiff = std::max( maxDiff, dist( b, _oldBeliefsV[i], DISTLINF ) );
        //    _oldBeliefsV[i] = b;
        //}
        //for( size_t I = 0; I < nrFactors(); ++I ) {
        //    Factor b( beliefF(I) );
        //    maxDiff = std::max( maxDiff, dist( b, _oldBeliefsF[I], DISTLINF ) );
        //    _oldBeliefsF[I] = b;
        //}
        maxDiff = accMaxDiff.reduce();

        if( props.verbose >= 3 )
            cerr << name() << "::run:  maxdiff " << maxDiff << " after " << _iters+1 << " passes" << endl;
    }

    if( maxDiff > _maxdiff )
        _maxdiff = maxDiff;

    if( props.verbose >= 1 ) {
        if( maxDiff > props.tol ) {
            if( props.verbose == 1 )
                cerr << endl;
                cerr << name() << "::run:  WARNING: not converged after " << _iters << " passes (" << toc() - tic << " seconds)...final maxdiff:" << maxDiff << endl;
        } else {
            if( props.verbose >= 3 )
                cerr << name() << "::run:  ";
                cerr << "converged in " << _iters << " passes (" << toc() - tic << " seconds)." << endl;
        }
    }

    return maxDiff;
}


void BP::calcBeliefV( size_t i, Prob &p ) const {
    p = Prob( var(i).states(), props.logdomain ? 0.0 : 1.0 );
    diaforeach( const Neighbor &I, nbV(i) )
        if( props.logdomain )
            p += newMessage( i, I.iter );
        else
            p *= newMessage( i, I.iter );
}


Factor BP::beliefV( size_t i ) const {
    Prob p;
    calcBeliefV( i, p );

    if( props.logdomain ) {
        p -= p.max();
        p.takeExp();
    }
    p.normalize();

    return( Factor( var(i), p ) );
}


Factor BP::beliefF( size_t I ) const {
    Prob p;
    calcBeliefF( I, p );

    if( props.logdomain ) {
        p -= p.max();
        p.takeExp();
    }
    p.normalize();

    return( Factor( factor(I).vars(), p ) );
}


vector<Factor> BP::beliefs() const {
    vector<Factor> result;
    for( size_t i = 0; i < nrVars(); ++i )
        result.push_back( beliefV(i) );
    for( size_t I = 0; I < nrFactors(); ++I )
        result.push_back( beliefF(I) );
    return result;
}


Factor BP::belief( const VarSet &ns ) const {
    if( ns.size() == 0 )
        return Factor();
    else if( ns.size() == 1 )
        return beliefV( findVar( *(ns.begin() ) ) );
    else {
        size_t I;
        for( I = 0; I < nrFactors(); I++ )
            if( factor(I).vars() >> ns )
                break;
        if( I == nrFactors() )
            DAI_THROW(BELIEF_NOT_AVAILABLE);
        return beliefF(I).marginal(ns);
    }
}


Real BP::logZ() const {
    Real sum = 0.0;
    for( size_t i = 0; i < nrVars(); ++i )
        sum += (1.0 - nbV(i).size()) * beliefV(i).entropy();
    for( size_t I = 0; I < nrFactors(); ++I )
        sum -= dist( beliefF(I), factor(I), DISTKL );
    return sum;
}


void BP::init( const VarSet &ns ) {
    for( VarSet::const_iterator n = ns.begin(); n != ns.end(); ++n ) {
        size_t ni = findVar( *n );
        diaforeach( const Neighbor &I, nbV( ni ) ) {
            Real val = props.logdomain ? 0.0 : 1.0;
            message( ni, I.iter ).fill( val );
            newMessage( ni, I.iter ).fill( val );
            if( props.updates == Properties::UpdateType::SEQMAX ) {
                updateResidual( ni, I.iter, 0.0 );
            }
        }
    }
    _iters = 0;
}


void BP::updateMessage( size_t i, size_t _I ) {
    if( recordSentMessages )
        _sentMessages.push_back(make_pair(i,_I));
    if( props.damping == 0.0 ) {
        message(i,_I) = newMessage(i,_I);
        if( props.updates == Properties::UpdateType::SEQMAX )
            updateResidual( i, _I, 0.0 );
    } else {
        if( props.logdomain )
            message(i,_I) = (message(i,_I) * props.damping) + (newMessage(i,_I) * (1.0 - props.damping));
        else
            message(i,_I) = (message(i,_I) ^ props.damping) * (newMessage(i,_I) ^ (1.0 - props.damping));
        if( props.updates == Properties::UpdateType::SEQMAX )
            updateResidual( i, _I, dist( newMessage(i,_I), message(i,_I), DISTLINF ) );
    }
}


void BP::updateResidual( size_t i, size_t _I, Real r ) {
    EdgeProp* pEdge = &_edges[i][_I];
    pEdge->residual = r;

    // rearrange look-up table (delete and reinsert new key)
    _lut.erase( _edge2lut[i][_I] );
    _edge2lut[i][_I] = _lut.insert( make_pair( r, make_pair(i, _I) ) );
}


} // end of namespace dai
