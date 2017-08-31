/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines class BP, which implements (Loopy) Belief Propagation
/// \todo Consider using a priority_queue for maximum residual schedule


#ifndef __defined_libdai_bp_h
#define __defined_libdai_bp_h


#include <string>
#include <dai/daialg.h>
#include <dai/factorgraph.h>
#include <dai/properties.h>
#include <dai/enum.h>
#include "Galois/UserContext.h"
#include "Galois/CheckedObject.h"
#include "Galois/Accumulator.h"

namespace dai {


/// Approximate inference algorithm "(Loopy) Belief Propagation"
/** The Loopy Belief Propagation algorithm uses message passing
 *  to approximate marginal probability distributions ("beliefs") for variables
 *  and factors (more precisely, for the subset of variables depending on the factor).
 *  There are two variants, the sum-product algorithm (corresponding to 
 *  finite temperature) and the max-product algorithm (corresponding to 
 *  zero temperature).
 *
 *  The messages \f$m_{I\to i}(x_i)\f$ are passed from factors \f$I\f$ to variables \f$i\f$. 
 *  In case of the sum-product algorith, the update equation is: 
 *    \f[ m_{I\to i}(x_i) \propto \sum_{x_{N_I\setminus\{i\}}} f_I(x_I) \prod_{j\in N_I\setminus\{i\}} \prod_{J\in N_j\setminus\{I\}} m_{J\to j}\f]
 *  and in case of the max-product algorithm:
 *    \f[ m_{I\to i}(x_i) \propto \max_{x_{N_I\setminus\{i\}}} f_I(x_I) \prod_{j\in N_I\setminus\{i\}} \prod_{J\in N_j\setminus\{I\}} m_{J\to j}\f]
 *  In order to improve convergence, the updates can be damped. For improved numerical stability,
 *  the updates can be done in the log-domain alternatively.
 *
 *  After convergence, the variable beliefs are calculated by:
 *    \f[ b_i(x_i) \propto \prod_{I\in N_i} m_{I\to i}(x_i)\f]
 *  and the factor beliefs are calculated by:
 *    \f[ b_I(x_I) \propto f_I(x_I) \prod_{j\in N_I} \prod_{J\in N_j\setminus\{I\}} m_{J\to j}(x_j) \f]
 *  The logarithm of the partition sum is calculated by:
 *    \f[ \log Z = \sum_i (1 - |N_i|) \sum_{x_i} b_i(x_i) \log b_i(x_i) - \sum_I \sum_{x_I} b_I(x_I) \log \frac{b_I(x_I)}{f_I(x_I)} \f]
 *
 *  For the max-product algorithm, a heuristic way of finding the MAP state (the 
 *  joint configuration of all variables which has maximum probability) is provided
 *  by the findMaximum() method, which can be called after convergence.
 *
 *  \note There are two implementations, an optimized one (the default) which caches IndexFor objects,
 *  and a slower, less complicated one which is easier to maintain/understand. The slower one can be 
 *  enabled by defining DAI_BP_FAST as false in the source file.
 */
class BP : public DAIAlgFG {
    protected:
        /// Type used for index cache
        typedef std::vector<size_t> ind_t;

        /// Type used for storing edge properties
        struct EdgeProp {
            /// Index cached for this edge
            ind_t  index;
            /// Old message living on this edge
            Prob   message;
            /// New message living on this edge
            Prob   newMessage;
            /// Residual for this edge
            Real   residual;
        };

        struct Task {
          size_t i, _I;
          int round;
          Real dist;
          Task() { }
          Task(size_t ii, size_t _II, int r, Real dd): i(ii), _I(_II), round(r), dist(dd) { }
          bool operator<(const Task& t) const {
            return dist < t.dist;
          }

	  int getID() const { return i ^ _I; }
        };

        struct EdgeData {
          int round;
          Galois::GChecked<char> lock;
          EdgeData(): round(0), lock(0) { }
        };

        struct Indexer: public std::unary_function<const Task&,unsigned int> {
          unsigned int operator()(const Task& t) const {
            const unsigned int maxvalue = 100000;
            float value = maxvalue + t.dist/2 * maxvalue;
            //std::cout << value << " " << t.dist << "\n";
            if (value > maxvalue) {
              return maxvalue;
            }
            if (value < 0) {
              return 0;
            }
            return value;
          }
        };

        struct TaskLess: public std::binary_function<const Task&,const Task&,bool> {
          bool operator()(const Task& a, const Task& b) {
            return a.dist < b.dist;
          }
        };

        struct TaskGreater: public std::binary_function<const Task&,const Task&,bool> {
          bool operator()(const Task& a, const Task& b) {
            return a.dist > b.dist;
          }
        };

        struct Process {
          typedef int tt_needs_parallel_break;
          BP& parent;
          std::vector<std::vector<EdgeData> >& edgeData;
          unsigned& count;

          Process(BP& p, std::vector<std::vector<EdgeData> >& e, unsigned& c): parent(p), edgeData(e), count(c) { }
          void operator()(const Task& t, Galois::UserContext<Task>& ctx) {
            parent.runProcess(t, edgeData, count, ctx);
          }
        };

        struct ComputeDiffs {
          typedef int tt_does_not_need_stats;
          typedef int tt_does_not_need_push;
          BP& parent;
          Galois::GReduceMax<Real>& accMaxDiff;
          ComputeDiffs(BP& p, Galois::GReduceMax<Real>& a): parent(p), accMaxDiff(a) { }
          void operator()(size_t v, Galois::UserContext<size_t>& ctx) {
            parent.runComputeDiffs(v, accMaxDiff);
          }
        };

        struct Initialize {
          typedef int tt_does_not_need_stats;
          typedef int tt_does_not_need_push;
          BP& parent;
          std::vector<Task>& initial;
          std::vector<std::vector<EdgeData> >& edgeData;
          Initialize(BP& p, std::vector<Task>& i, 
              std::vector<std::vector<EdgeData> >& e): parent(p), initial(i), edgeData(e) { }
          void operator()(size_t index, Galois::UserContext<size_t>& ctx) {
            parent.runInitialize(index, initial, edgeData);
          }
        };

        /// Stores all edge properties
        std::vector<std::vector<EdgeProp> > _edges;
        /// Type of lookup table (only used for maximum-residual BP)
        typedef std::multimap<Real, std::pair<std::size_t, std::size_t> > LutType;
        /// Lookup table (only used for maximum-residual BP)
        std::vector<std::vector<LutType::iterator> > _edge2lut;
        /// Lookup table (only used for maximum-residual BP)
        LutType _lut;
        /// Maximum difference between variable beliefs encountered so far
        Real _maxdiff;
        /// Number of iterations needed
        size_t _iters;
        /// The history of message updates (only recorded if \a recordSentMessages is \c true)
        std::vector<std::pair<std::size_t, std::size_t> > _sentMessages;
        /// Stores variable beliefs of previous iteration
        std::vector<Factor> _oldBeliefsV;
        /// Stores factor beliefs of previous iteration
        std::vector<Factor> _oldBeliefsF;
        /// Stores the update schedule
        std::vector<Edge> _updateSeq;

    public:
        /// Parameters for BP
        struct Properties {
            /// Enumeration of possible update schedules
            /** The following update schedules have been defined:
             *  - PARALL parallel updates
             *  - SEQFIX sequential updates using a fixed sequence
             *  - SEQRND sequential updates using a random sequence
             *  - SEQMAX maximum-residual updates [\ref EMK06]
             */
            DAI_ENUM(UpdateType,SEQFIX,SEQRND,SEQMAX,PARALL,SEQPRI,SEQPRIASYNC);

            /// Enumeration of inference variants
            /** There are two inference variants:
             *  - SUMPROD Sum-Product
             *  - MAXPROD Max-Product (equivalent to Min-Sum)
             */
            DAI_ENUM(InfType,SUMPROD,MAXPROD);

            /// Verbosity (amount of output sent to stderr)
            size_t verbose;

            /// Maximum number of iterations
            size_t maxiter;

            /// Maximum time (in seconds)
            double maxtime;

            /// Tolerance for convergence test
            Real tol;

            /// Whether updates should be done in logarithmic domain or not
            bool logdomain;

            /// Damping constant (0.0 means no damping, 1.0 is maximum damping)
            Real damping;

            /// Message update schedule
            UpdateType updates;

            /// Inference variant
            InfType inference;

            std::string worklist;
        } props;

        /// Specifies whether the history of message updates should be recorded
        bool recordSentMessages;

    public:
    /// \name Constructors/destructors
    //@{
        /// Default constructor
        BP() : DAIAlgFG(), _edges(), _edge2lut(), _lut(), _maxdiff(0.0), _iters(0U), _sentMessages(), _oldBeliefsV(), _oldBeliefsF(), _updateSeq(), props(), recordSentMessages(false) {}

        /// Construct from FactorGraph \a fg and PropertySet \a opts
        /** \param fg Factor graph.
         *  \param opts Parameters @see Properties
         */
        BP( const FactorGraph & fg, const PropertySet &opts ) : DAIAlgFG(fg), _edges(), _maxdiff(0.0), _iters(0U), _sentMessages(), _oldBeliefsV(), _oldBeliefsF(), _updateSeq(), props(), recordSentMessages(false) {
            setProperties( opts );
            construct();
        }

        /// Copy constructor
        BP( const BP &x ) : DAIAlgFG(x), _edges(x._edges), _edge2lut(x._edge2lut), _lut(x._lut), _maxdiff(x._maxdiff), _iters(x._iters), _sentMessages(x._sentMessages), _oldBeliefsV(x._oldBeliefsV), _oldBeliefsF(x._oldBeliefsF), _updateSeq(x._updateSeq), props(x.props), recordSentMessages(x.recordSentMessages) {
            for( LutType::iterator l = _lut.begin(); l != _lut.end(); ++l )
                _edge2lut[l->second.first][l->second.second] = l;
        }

        /// Assignment operator
        BP& operator=( const BP &x ) {
            if( this != &x ) {
                DAIAlgFG::operator=( x );
                _edges = x._edges;
                _lut = x._lut;
                for( LutType::iterator l = _lut.begin(); l != _lut.end(); ++l )
                    _edge2lut[l->second.first][l->second.second] = l;
                _maxdiff = x._maxdiff;
                _iters = x._iters;
                _sentMessages = x._sentMessages;
                _oldBeliefsV = x._oldBeliefsV;
                _oldBeliefsF = x._oldBeliefsF;
                _updateSeq = x._updateSeq;
                props = x.props;
                recordSentMessages = x.recordSentMessages;
            }
            return *this;
        }
    //@}

    /// \name General InfAlg interface
    //@{
        virtual BP* clone() const { return new BP(*this); }
        virtual BP* construct( const FactorGraph &fg, const PropertySet &opts ) const { return new BP( fg, opts ); }
        virtual std::string name() const { return "BP"; }
        virtual Factor belief( const Var &v ) const { return beliefV( findVar( v ) ); }
        virtual Factor belief( const VarSet &vs ) const;
        virtual Factor beliefV( size_t i ) const;
        virtual Factor beliefF( size_t I ) const;
        virtual std::vector<Factor> beliefs() const;
        virtual Real logZ() const;
        /** \pre Assumes that run() has been called and that \a props.inference == \c MAXPROD
         */
        std::vector<std::size_t> findMaximum() const { return dai::findMaximum( *this ); }
        virtual void init();
        virtual void init( const VarSet &ns );
        virtual Real run();
        virtual Real maxDiff() const { return _maxdiff; }
        virtual size_t Iterations() const { return _iters; }
        virtual void setMaxIter( size_t maxiter ) { props.maxiter = maxiter; }
        virtual void setProperties( const PropertySet &opts );
        virtual PropertySet getProperties() const;
        virtual std::string printProperties() const;
    //@}

    /// \name Additional interface specific for BP
    //@{
        /// Returns history of which messages have been updated
        const std::vector<std::pair<std::size_t, std::size_t> >& getSentMessages() const {
            return _sentMessages;
        }

        /// Clears history of which messages have been updated
        void clearSentMessages() { _sentMessages.clear(); }
    //@}

    protected:
        /// Returns constant reference to message from the \a _I 'th neighbor of variable \a i to variable \a i
        const Prob & message(size_t i, size_t _I) const { return _edges[i][_I].message; }
        /// Returns reference to message from the \a _I 'th neighbor of variable \a i to variable \a i
        Prob & message(size_t i, size_t _I) { return _edges[i][_I].message; }
        /// Returns constant reference to updated message from the \a _I 'th neighbor of variable \a i to variable \a i
        const Prob & newMessage(size_t i, size_t _I) const { return _edges[i][_I].newMessage; }
        /// Returns reference to updated message from the \a _I 'th neighbor of variable \a i to variable \a i
        Prob & newMessage(size_t i, size_t _I) { return _edges[i][_I].newMessage; }
        /// Returns constant reference to cached index for the edge between variable \a i and its \a _I 'th neighbor
        const ind_t & index(size_t i, size_t _I) const { return _edges[i][_I].index; }
        /// Returns reference to cached index for the edge between variable \a i and its \a _I 'th neighbor
        ind_t & index(size_t i, size_t _I) { return _edges[i][_I].index; }
        /// Returns constant reference to residual for the edge between variable \a i and its \a _I 'th neighbor
        const Real & residual(size_t i, size_t _I) const { return _edges[i][_I].residual; }
        /// Returns reference to residual for the edge between variable \a i and its \a _I 'th neighbor
        Real & residual(size_t i, size_t _I) { return _edges[i][_I].residual; }
        void runProcess(const Task&, std::vector<std::vector<EdgeData> >&, unsigned&, Galois::UserContext<Task>&);
        void runComputeDiffs(size_t v, Galois::GReduceMax<Real>& acc);
        void runInitialize(size_t, std::vector<Task>&, std::vector<std::vector<EdgeData> >& edgeData);

        /// Calculate the product of factor \a I and the incoming messages
        /** If \a without_i == \c true, the message coming from variable \a i is omitted from the product
         *  \note This function is used by calcNewMessage() and calcBeliefF()
         */
        virtual Prob calcIncomingMessageProduct( size_t I, bool without_i, size_t i ) const;
        /// Calculate the updated message from the \a _I 'th neighbor of variable \a i to variable \a i
        virtual void calcNewMessage( size_t i, size_t _I );
        /// Replace the "old" message from the \a _I 'th neighbor of variable \a i to variable \a i by the "new" (updated) message
        void updateMessage( size_t i, size_t _I );
        /// Set the residual (difference between new and old message) for the edge between variable \a i and its \a _I 'th neighbor to \a r
        void updateResidual( size_t i, size_t _I, Real r );
        /// Finds the edge which has the maximum residual (difference between new and old message)
        void findMaxResidual( size_t &i, size_t &_I );
        /// Calculates unnormalized belief of variable \a i
        virtual void calcBeliefV( size_t i, Prob &p ) const;
        /// Calculates unnormalized belief of factor \a I
        virtual void calcBeliefF( size_t I, Prob &p ) const {
            p = calcIncomingMessageProduct( I, false, 0 );
        }

        /// Helper function for constructors
        virtual void construct();
};


} // end of namespace dai


#endif
