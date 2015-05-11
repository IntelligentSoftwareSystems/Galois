/** Distributed LC InOut Graph -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Gurbinder Gill <gill@cs.utexas.edu>
 */

#ifndef GALOIS_GRAPH_LC_DIST_INOUT_H
#define GALOIS_GRAPH_LC_DIST_INOUT_H

#include <vector>
#include <iostream>

#include "Galois/Runtime/Context.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Runtime/Serialize.h"
#include "Galois/Runtime/PerHostStorage.h"
#include "Galois/Runtime/DistSupport.h"


namespace Galois {
  namespace Graph {

    template<typename NodeTy, typename EdgeTy>
      class LC_Dist_InOut {

        struct EdgeImplTy;

        struct NodeImplTy :public Runtime::Lockable {
          NodeImplTy(EdgeImplTy* start, unsigned len, EdgeImplTy* In_start, unsigned len_inEdges) :b(start), e(start), len(len), b_inEdges(In_start), e_inEdges(In_start), len_inEdges(len_inEdges), remote(false)
          {}
          ~NodeImplTy() {
            if (remote) {
              delete[] b;
              delete[] b_inEdges;
            }
            b = e = nullptr;
            b_inEdges = e_inEdges = nullptr;
          }

          //Serialization support
          typedef int tt_has_serialize;
          NodeImplTy(Runtime::DeSerializeBuffer& buf) :remote(true) {
            ptrdiff_t diff;
            ptrdiff_t diff_inEdges;
            gDeserialize(buf, data, len, diff, len_inEdges, diff_inEdges);
            b = e = len ? (new EdgeImplTy[len]) : nullptr;
            b_inEdges = e_inEdges = len_inEdges ? (new EdgeImplTy[len_inEdges]) : nullptr;
            while (diff--) {
              Runtime::gptr<NodeImplTy> ptr;
              gDeserialize(buf, ptr);
              append(ptr);
            }

            while (diff_inEdges--) {
              Runtime::gptr<NodeImplTy> ptr;
              gDeserialize(buf, ptr);
              append_inEdges(ptr);

            }
          }

          void serialize(Runtime::SerializeBuffer& buf) const {
            EdgeImplTy* begin = b;
            EdgeImplTy* end = e;

            EdgeImplTy* begin_In = b_inEdges;
            EdgeImplTy* end_In = e_inEdges;

            ptrdiff_t diff = end - begin;
            ptrdiff_t diff_inEdges = end_In - begin_In;
            gSerialize(buf, data, len, diff, len_inEdges, diff_inEdges);
            while (begin != end) {
              gSerialize(buf, begin->dst);
              ++begin;
            }
            while (begin_In != end_In) {
              gSerialize(buf, begin_In->dst);
              ++begin_In;
            }

          }
          void deserialize(Runtime::DeSerializeBuffer& buf) {
            assert(!remote);
            ptrdiff_t diff;
            unsigned _len;
            ptrdiff_t diff_inEdges;
            unsigned _len_inEdges;

            EdgeImplTy* begin = b;
            EdgeImplTy* begin_In = b_inEdges;
            gDeserialize(buf, data, _len, diff, _len_inEdges, diff_inEdges);
            assert(_len == len);
            assert(diff >= e - b);
            // inEdges
            assert(_len_inEdges == len_inEdges);
            assert(diff_inEdges >= e_inEdges - b_inEdges);
            while (diff--) {
              gDeserialize(buf, begin->dst);
              ++begin;
            }
            e = begin;
            while (diff_inEdges--) {
              gDeserialize(buf, begin_In->dst);
              ++begin_In;
            }
            e_inEdges = begin_In;
          }

          NodeTy data;
          EdgeImplTy* b;
          EdgeImplTy* e;
          unsigned len;
          // InEdge list for each node
          EdgeImplTy* b_inEdges;
          EdgeImplTy* e_inEdges;
          unsigned len_inEdges;
          //
          bool remote;

          EdgeImplTy* append(Runtime::gptr<NodeImplTy> dst) {
            assert(e-b < len);
            e->dst = dst;
            return e++;
          }

          // Appending In_Edges
          EdgeImplTy* append_inEdges(Runtime::gptr<NodeImplTy> dst) {
            assert(e_inEdges-b_inEdges < len_inEdges);
            e_inEdges->dst = dst;
            return e_inEdges++;
          }

          unsigned get_num_outEdges()
          {
            return len;
          }
          unsigned get_num_inEdges()
          {
            return len_inEdges;
          }


          EdgeImplTy* begin() { return b; }
          EdgeImplTy* end() { return e; }

          // for In_edge list for each node 
          EdgeImplTy* begin_inEdges() { return b_inEdges; }
          EdgeImplTy* end_inEdges() { return e_inEdges; }

        };

        struct EdgeImplTy {
          Runtime::gptr<NodeImplTy> dst;
        };

        public:

        typedef Runtime::gptr<NodeImplTy> GraphNode;
        //G
        template<bool _has_id>
          struct with_id { typedef LC_Dist_InOut type; };

        template<typename _node_data>
          struct with_node_data { typedef LC_Dist_InOut<_node_data, EdgeTy> type; };

        template<typename _edge_data>
          struct with_edge_data { typedef LC_Dist_InOut<NodeTy, _edge_data> type; };


        private:

        typedef std::vector<NodeImplTy> NodeData;
        typedef std::vector<EdgeImplTy> EdgeData;
        NodeData Nodes;
        EdgeData Edges;
        EdgeData In_Edges;

        std::vector<std::pair<GraphNode, std::atomic<int> > > Starts;
        std::vector<unsigned> Num;
        std::vector<unsigned> PrefixNum;

        Runtime::PerHost<LC_Dist_InOut> self;

        friend class Runtime::PerHost<LC_Dist_InOut>;

        LC_Dist_InOut(Runtime::PerHost<LC_Dist_InOut> _self, std::vector<unsigned>& edges, std::vector<unsigned>& In_edges) 
          :Starts(Runtime::NetworkInterface::Num),
          Num(Runtime::NetworkInterface::Num),
          PrefixNum(Runtime::NetworkInterface::Num),
          self(_self)
        {
          //std::cerr << Runtime::NetworkInterface::ID << " Construct\n";
          Runtime::trace("LC_Dist_InOut with % nodes total\n", edges.size());
          //Fill up metadata vectors
          for (unsigned h = 0; h < Runtime::NetworkInterface::Num; ++h) {
            auto p = block_range(edges.begin(), edges.end(), h,
                Runtime::NetworkInterface::Num);
            Num[h] = std::distance(p.first, p.second);
          }
          std::partial_sum(Num.begin(), Num.end(), PrefixNum.begin());

          // std::copy(Num.begin(), Num.end(), std::ostream_iterator<unsigned>(std::cout, ","));
          // std::cout << "\n";
          // std::copy(PrefixNum.begin(), PrefixNum.end(), std::ostream_iterator<unsigned>(std::cout, ","));
          // std::cout << "\n";

          //Block nodes
          auto p = block_range(edges.begin(), edges.end(), 
              Runtime::NetworkInterface::ID,
              Runtime::NetworkInterface::Num);
          //Block in_edge vector
          auto p_inEdges = block_range(In_edges.begin(), In_edges.end(),
              Runtime::NetworkInterface::ID,
              Runtime::NetworkInterface::Num);

          //NOTE: range of p and p_inEdges will be same since the size(edges) == size(In_edges)
          //Allocate Edges
          unsigned sum = std::accumulate(p.first, p.second, 0); // Total number of egdes on this host (me)
          Edges.resize(sum);
          EdgeImplTy* cur = &Edges[0];

          //Allocate In_Edges
          unsigned sum_inEdges = std::accumulate(p_inEdges.first, p_inEdges.second, 0);
          In_Edges.resize(sum_inEdges);
          EdgeImplTy* cur_In = &In_Edges[0]; //cur edgelist for in_edges

          //allocate Nodes
          Nodes.reserve(std::distance(p.first, p.second));
          //std::cout << " I am : " <<Runtime::NetworkInterface::ID << " with Nodes = " << std::distance(p.first, p.second)<<"\n";
          for (auto ii = p.first, ii_in = p_inEdges.first; ii != p.second; ++ii,++ii_in) {
            Nodes.emplace_back(cur, *ii, cur_In, *ii_in);
            cur += *ii;
            cur_In += *ii_in;
          }
          Starts[Runtime::NetworkInterface::ID].first = GraphNode(&Nodes[0]);
          Starts[Runtime::NetworkInterface::ID].second = 2;
        }

        ~LC_Dist_InOut() {
          auto ii = iterator(this, Runtime::NetworkInterface::ID == 0 ? 0 : PrefixNum[Runtime::NetworkInterface::ID - 1] );
          auto ee = iterator(this, PrefixNum[Runtime::NetworkInterface::ID] );
          for (; ii != ee; ++ii)
            acquireNode(*ii, MethodFlag::ALL);
        }

        static void getStart(Runtime::PerHost<LC_Dist_InOut> graph, uint32_t whom) {
          //std::cerr << Runtime::NetworkInterface::ID << " getStart " << whom << "\n";
          Runtime::getSystemNetworkInterface().sendAlt(whom, putStart, graph, Runtime::NetworkInterface::ID, graph->Starts[Runtime::NetworkInterface::ID].first);
        }

        static void putStart(Runtime::PerHost<LC_Dist_InOut> graph, uint32_t whom, GraphNode start) {
          graph->Starts[whom].first = start;
          graph->Starts[whom].second = 2;
        }

        //blocking
        void fillStarts(uint32_t host) {
          if (Starts[host].second != 2) {
            int ex = 0;
            unsigned id = Runtime::NetworkInterface::ID;
            bool swap = Starts[host].second.compare_exchange_strong(ex,1);
            if (swap)
              Runtime::getSystemNetworkInterface().sendAlt(host, getStart, self, Runtime::NetworkInterface::ID);
            while (Starts[host].second != 2) { Runtime::doNetworkWork(); }
          }
          assert(Starts[host].first);
        }

        GraphNode generateNodePtr(unsigned x) {
          auto ii = std::upper_bound(PrefixNum.begin(), PrefixNum.end(), x);
          assert(ii != PrefixNum.end());
          unsigned host = std::distance(PrefixNum.begin(), ii);
          unsigned offset = x - (host == 0 ? 0 : *(ii - 1));
          fillStarts(host);
          return Starts[host].first + offset;
        }

        void acquireNode(Runtime::gptr<NodeImplTy> node, Galois::MethodFlag mflag) {
          acquire(node, mflag);
        }

        public:
        typedef EdgeImplTy edge_data_type;
        typedef NodeImplTy node_data_type;
        typedef typename EdgeData::reference edge_data_reference;
        typedef typename NodeData::reference node_data_reference;


        //This is technically a const (non-mutable) iterator
        class iterator : public std::iterator<std::random_access_iterator_tag,
        GraphNode, ptrdiff_t, GraphNode, GraphNode> {
          LC_Dist_InOut* g;
          unsigned x;

          friend class LC_Dist_InOut;
          iterator(LC_Dist_InOut* _g, unsigned _x) :g(_g), x(_x) {}

          public:
          iterator() :g(nullptr), x(0) {}
          iterator(const iterator&) = default;
          bool operator==(const iterator& rhs) const { return g == rhs.g & x == rhs.x; }
          bool operator!=(const iterator& rhs) const { return !(*this == rhs); }
          bool operator< (const iterator& rhs) const { return x < rhs.x; }
          bool operator> (const iterator& rhs) const { return x > rhs.x; }
          bool operator<=(const iterator& rhs) const { return x <= rhs.x; }
          bool operator>=(const iterator& rhs) const { return x >= rhs.x; }

          GraphNode operator*() { return g->generateNodePtr(x); }
          GraphNode operator->() { return g->generateNodePtr(x); }
          GraphNode operator[](int n) { return g->generateNodePtr(x + n); }

          iterator& operator++()    { ++x; return *this; }
          iterator& operator--()    { --x; return *this; }
          iterator  operator++(int) { auto old = *this; ++x; return old; }
          iterator  operator--(int) { auto old = *this; --x; return old; }
          iterator  operator+(int n) { auto ret = *this; ret.x += n; return ret; }
          iterator  operator-(int n) { auto ret = *this; ret.x -= n; return ret; }
          iterator& operator+=(int n) { x += n; return *this; }
          iterator& operator-=(int n) { x -= n; return *this; }
          ptrdiff_t operator-(const iterator& rhs) { return x - rhs.x; }
          //Trivially copyable
          typedef int tt_is_copyable;
        };

        typedef iterator local_iterator;
        typedef EdgeImplTy* edge_iterator;

        //! Creation and destruction
        typedef Runtime::PerHost<LC_Dist_InOut> pointer;
        static pointer allocate(std::vector<unsigned>& edges, std::vector<unsigned>& In_edges) {
          return pointer::allocate(edges, In_edges);
        }
        static void deallocate(pointer ptr) {
          pointer::deallocate(ptr);
        }

        //! Mutation

        template<typename... Args>
          edge_iterator  addEdge(GraphNode src, GraphNode dst, MethodFlag mflag = MethodFlag::ALL) {
          acquire(src, mflag);
          return src->append(dst);
        }

        // Adding inEdges to the Graph.
        template<typename... Args>
        edge_iterator addInEdge(GraphNode src, GraphNode dst, MethodFlag mflag = MethodFlag::ALL) {
          acquire(src, mflag);
          return src->append_inEdges(dst);
        }
        //! Access

        NodeTy& at(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
          acquire(N, mflag);
          return N->data;
        }

        unsigned get_num_outEdges(GraphNode N)
        {
          return N->get_num_outEdges();
        }

        //EdgeTy& at(edge_iterator E, MethodFlag mflag = MethodFlag::ALL) {
          //return E->data;
        //}

        GraphNode dst(edge_iterator E, MethodFlag mflag = MethodFlag::ALL) {
          return E->dst;
        }

        //! Capacity
        unsigned size() {
          return *PrefixNum.rbegin();
        }

        //! Iterators

        iterator begin() { return iterator(this, 0); }
        iterator end  () { return iterator(this, PrefixNum[Runtime::NetworkInterface::Num - 1]); }

        local_iterator local_begin() {
          if (Runtime::LL::getTID() == 0)
            return iterator(this, Runtime::NetworkInterface::ID == 0 ? 0 : PrefixNum[Runtime::NetworkInterface::ID - 1] );
          else
            return local_end();
        }
        local_iterator local_end  () { return iterator(this, PrefixNum[Runtime::NetworkInterface::ID]); }

        edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
          acquire(N, mflag);
          if (mflag != MethodFlag::SRC_ONLY && mflag != MethodFlag::NONE)
          {
            /** prefetch **/
            for (auto ii = N->begin(), ee = N->end(); ii != ee; ++ii) {
              prefetch(ii->dst);
            }
            for (edge_iterator ii = N->begin(), ee = N->end(); ii != ee; ++ii) {
              acquireNode(ii->dst, mflag);
            }
          }
          return N->begin();
        }

        edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
          acquireNode(N, mflag);
          return N->end();
        }


        /**
         * In Edge iterators 
         *
         */

        edge_iterator in_edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
          acquire(N, mflag);
          if (mflag != MethodFlag::SRC_ONLY && mflag != MethodFlag::NONE)
          {
            /** prefetch **/
            for (auto ii = N->begin_inEdges(), ee = N->end_inEdges(); ii != ee; ++ii) {
              prefetch(ii->dst);
            }
            for (edge_iterator ii = N->begin_inEdges(), ee = N->end_inEdges(); ii !=ee; ++ii) {
              acquireNode(ii->dst, mflag);	
            }
          }
          return N->begin_inEdges(); 
        }

        edge_iterator in_edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
          acquireNode(N, mflag);
          return N->end_inEdges();  
        }

        template<typename Compare>
          void sort_edges(GraphNode N, Compare comp, MethodFlag mflag = MethodFlag::ALL) {
            std::sort(edge_begin(N, mflag), edge_end(N, mflag),
                [&comp] (const EdgeImplTy& e1, const EdgeImplTy& e2) {
                return comp(e1.dst, e1.data, e2.dst, e2.data);
                });
          }

        /*** Additional functions ***/
        void prefetch_all()
        {
          for(auto ii = this->begin(); ii != this->end(); ++ii)
          {
            prefetch(*ii);
          }
        }

        void prefetch_these(unsigned start_index, unsigned end_index)
        {
          auto ii = this->begin() +  start_index;
          auto ei = this->begin() + end_index;
          for(; ii != ei; ++ii)
          {
            prefetch(*ii);
          }
        }


        /*
         * Per host graph constructions
         * Each host constructs edges for their nodes.
         */

        typedef Runtime::SerializeBuffer SendBuffer;
        typedef Runtime::DeSerializeBuffer RecvBuffer;

        struct createOnEachHost_Edges {
          pointer g;
          std::string inputFile;
          Galois::Graph::FileGraph fg;
          std::vector<unsigned> counts;
          std::vector<unsigned> In_counts;

          createOnEachHost_Edges() {}
          createOnEachHost_Edges(pointer _g, std::string _inputFile):g(_g), inputFile(_inputFile){}

          void static go(pointer _g, std::string _inputFile, unsigned total_nodes)
          {
            std::cout << "inside function\n";
            unsigned h = 0;
            for(auto N = _g->begin(); N != _g->end(); ++N)
            {
              h++;
            }
            std::cout << "H -> " << h <<"\n";
            Galois::for_each(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(total_nodes), createOnEachHost_Edges(_g, _inputFile), Galois::loopname("Construct outEdges"));
          }

          //void operator()(unsigned x, Galois::UserContext<unsigned>& cnx)
          void operator()(unsigned tid, unsigned)
          {
            unsigned id = Runtime::NetworkInterface::ID;
            unsigned num = Runtime::NetworkInterface::Num;

            //std::cout << " I m on HOST : " << id << " file name : " << inputFile << "\n";

            //XXX put lock on this
            if(fg.size() == 0 )
            {
              fg.fromFile(inputFile);

            for(auto& N : fg)
            {
              counts.push_back(std::distance(fg.edge_begin(N), fg.edge_end(N)));
              for(auto ii = fg.edge_begin(N), ei = fg.edge_end(N); ii != ei; ++ii)
              {
                unsigned dst = fg.getEdgeDst(ii);
                if(dst >= In_counts.size()) {
                  In_counts.resize(dst+1);
                }
                In_counts[dst]+=1;
              }
            }
            if(counts.size() >  In_counts.size())
              In_counts.resize(counts.size());
            }

          unsigned total_nodes = counts.size();
          unsigned zero = 0;
          unsigned start_index, end_index;
          std::tie(start_index, end_index) = Galois::block_range(zero, total_nodes, id, num);



            for(unsigned x = start_index ; x < end_index; ++x)
            {
              auto fgn = *(fg.begin() + x);
              auto gn = *(g->begin() + x);
              for (auto ii = fg.edge_begin(fgn), ee = fg.edge_end(fgn); ii != ee; ++ii) {
                unsigned dst = fg.getEdgeDst(ii);
                //int val = fg.getEdgeData<int>(ii);
                g->addEdge(gn, *(g->begin() + dst), Galois::MethodFlag::SRC_ONLY);
              }
            }
        }

          typedef int tt_has_serialize;
          void serialize(SendBuffer& buf) const { gSerialize(buf, g, inputFile); }
          void deserialize(RecvBuffer& buf) { gDeserialize(buf, g, inputFile); }
        };





        struct createOnEachHost_InEdges {
          pointer g;
          std::string inputFile_tr;
          Galois::Graph::FileGraph fg;
          std::vector<unsigned> counts;
          std::vector<unsigned> In_counts;

          createOnEachHost_InEdges() {}
          createOnEachHost_InEdges(pointer _g, std::string _inputFile_tr):g(_g), inputFile_tr(_inputFile_tr){}

          void static go(pointer _g, std::string _inputFile_tr, unsigned total_nodes)
          {

            Galois::for_each(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(total_nodes), createOnEachHost_InEdges(_g, _inputFile_tr), Galois::loopname("Construct InEdges"));
          }

          //void operator()(unsigned x, Galois::UserContext<unsigned>& cnx)
          void operator()(unsigned tid, unsigned)
          {
            unsigned id = Runtime::NetworkInterface::ID;
            unsigned num = Runtime::NetworkInterface::Num;


            //XXX put lock on this
            if(fg.size() == 0 )
            {
              fg.fromFile(inputFile_tr);

            for(auto& N : fg)
            {
              counts.push_back(std::distance(fg.edge_begin(N), fg.edge_end(N)));
              for(auto ii = fg.edge_begin(N), ei = fg.edge_end(N); ii != ei; ++ii)
              {
                unsigned dst = fg.getEdgeDst(ii);
                if(dst >= In_counts.size()) {
                  In_counts.resize(dst+1);
                }
                In_counts[dst]+=1;
              }
            }
            if(counts.size() >  In_counts.size())
              In_counts.resize(counts.size());
            }

          unsigned total_nodes = counts.size();
          unsigned zero = 0;
          unsigned start_index, end_index;
          std::tie(start_index, end_index) = Galois::block_range(zero, total_nodes, id, num);



          for(unsigned x = start_index ; x < end_index; ++x)
          {
            auto fgn = *(fg.begin() + x);
            auto gn = *(g->begin() + x);
            for (auto ii = fg.edge_begin(fgn), ee = fg.edge_end(fgn); ii != ee; ++ii) {
              unsigned dst = fg.getEdgeDst(ii);
              //int val = fg.getEdgeData<int>(ii);
              g->addInEdge(gn, *(g->begin() + dst), Galois::MethodFlag::SRC_ONLY);
            }
          }

        }

          typedef int tt_has_serialize;
          void serialize(SendBuffer& buf) const { gSerialize(buf, g, inputFile_tr); }
          void deserialize(RecvBuffer& buf) { gDeserialize(buf, g, inputFile_tr); }
        };

        /*
         * called on master host to initiate per host graph
         * construction
         *
         */
        template<typename... Args>
        static void createPerHost(pointer& g, std::string inputFile, std::string inputFile_tr, Args... args)
        {

          unsigned num = Runtime::NetworkInterface::Num;
          unsigned id = Runtime::NetworkInterface::ID;

          Galois::Graph::FileGraph fg;
          fg.fromFile(inputFile);


          std::vector<unsigned> counts;
          std::vector<unsigned> In_counts;
          for(auto& N : fg)
          {
            counts.push_back(std::distance(fg.edge_begin(N), fg.edge_end(N)));
            for(auto ii = fg.edge_begin(N), ei = fg.edge_end(N); ii != ei; ++ii)
            {
              unsigned dst = fg.getEdgeDst(ii);
              if(dst >= In_counts.size()) {
                // +1 is imp because vec.resize makes sure new vec can hold dst entries so it
                // will not have vec[dst] which is (dst+1)th entry!!
                In_counts.resize(dst+1);
              }
              In_counts[dst]+=1;
            }
          }
          if(counts.size() >  In_counts.size())
            In_counts.resize(counts.size());

          unsigned TOTAL_NODES = counts.size();

          std::cout << "TOTAL NODEs : " << TOTAL_NODES << "\n";
          g = allocate(counts, In_counts);
          counts = std::vector<unsigned>();
          In_counts = std::vector<unsigned>();
          counts.shrink_to_fit();
          In_counts.shrink_to_fit();
          std::cout << "Allocation complete\n";
          
          //Runtime::on_each(createOnEachHost{g, inputFile});
          Runtime::on_each_impl_dist(createOnEachHost_Edges(g, inputFile), "loop");
          Runtime::on_each_impl_dist(createOnEachHost_InEdges(g, inputFile_tr), "loop");
          //Galois::for_each(g->begin(), g->end(), createOnEachHost{g, inputFile}, Galois::loopname("For each"));
          //createOnEachHost_Edges::go(g, inputFile, TOTAL_NODES);
          //createOnEachHost_InEdges::go(g, inputFile_tr, TOTAL_NODES);

        }

      };

  } //namespace Graph
} //namespace Galois








        //template<typename... Args>
        //static void createOnHost(pointer g, std::string inputFile, Args... args)
        //{

          //unsigned num = Runtime::NetworkInterface::Num;
          //unsigned id = Runtime::NetworkInterface::ID;

          //Galois::Graph::FileGraph fg;
          //fg.fromFile(inputFile);
          //std::vector<unsigned> counts;
          //std::vector<unsigned> In_counts;
          //for(auto& N : fg)
          //{
            //counts.push_back(std::distance(fg.edge_begin(N), fg.edge_end(N)));
            //for(auto ii = fg.edge_begin(N), ei = fg.edge_end(N); ii != ei; ++ii)
            //{
              //unsigned dst = fg.getEdgeDst(ii);
              //if(dst >= In_counts.size()) {
                //// +1 is imp because vec.resize makes sure new vec can hold dst entries so it
                //// will not have vec[dst] which is (dst+1)th entry!!
                //In_counts.resize(dst+1);
              //}
              //In_counts[dst]+=1;
            //}
          //}
          //if(counts.size() >  In_counts.size())
            //In_counts.resize(counts.size());


          //unsigned total_nodes = counts.size();
          //unsigned zero = 0;
          //unsigned start_index, end_index;
          //std::tie(start_index, end_index) = Galois::block_range(zero, total_nodes, id, num);


         //std::cout << " ON: "<< Runtime::NetworkInterface::ID << " Nodes From [ " << start_index << " , " << end_index  << "]\n";
          ////std::cout << "Adding outEdges and Inedges on : " << Runtime::NetworkInterface::ID << "\n";
          //for (unsigned x = start_index; x < end_index; ++x) {
            //auto fgn = *(fg.begin() + x);
            //auto gn = *(g->begin() + x);
            ////std::cout << " x : " << x << " gn : " << gn << " \n";
            //for (auto ii = fg.edge_begin(fgn), ee = fg.edge_end(fgn); ii != ee; ++ii) {
              //unsigned dst = fg.getEdgeDst(ii);
              //auto gn_dst = *(g->begin() + dst);
              ////std::cout << gn_dst.isLocal() << "\n";
              ////std::cout << " dst : " << dst << " gdst : " << (g->begin() + dst) << " \t";
              ////int val = fg.getEdgeData<int>(ii);
              ////g->addEdge(gn, *(g->begin() + dst), Galois::MethodFlag::SRC_ONLY);
              ////if((dst >= start_index) && (dst < end_index) ) //XXX: check this: Is local
                ////g->addInEdge(*(g->begin() + dst),gn, Galois::MethodFlag::SRC_ONLY);
            //}
            
          //}
          ////std::cout << " DONE: edges on host : " << Runtime::NetworkInterface::ID << "\n";


        //}



   //static void createPerHost(pointer& g, std::string inputFile, Args... args)
        //{

          //unsigned num = Runtime::NetworkInterface::Num;
          //unsigned id = Runtime::NetworkInterface::ID;

          //Galois::Graph::FileGraph fg;
          //fg.fromFile(inputFile);
          //std::vector<unsigned> counts;
          //std::vector<unsigned> In_counts;
          //for(auto& N : fg)
          //{
            //counts.push_back(std::distance(fg.edge_begin(N), fg.edge_end(N)));
            //for(auto ii = fg.edge_begin(N), ei = fg.edge_end(N); ii != ei; ++ii)
            //{
              //unsigned dst = fg.getEdgeDst(ii);
              //if(dst >= In_counts.size()) {
                //// +1 is imp because vec.resize makes sure new vec can hold dst entries so it
                //// will not have vec[dst] which is (dst+1)th entry!!
                //In_counts.resize(dst+1);
              //}
              //In_counts[dst]+=1;
            //}
          //}
          //if(counts.size() >  In_counts.size())
            //In_counts.resize(counts.size());

          //unsigned TOTAL_NODES = counts.size();

          ////std::cout << "size of transpose : " << In_counts.size() <<" : : "<<In_counts[0] <<"\n";
          ////std::cout << "size of counts : " << counts.size() << "\n";
          //g = allocate(counts, In_counts);
          //std::cout << "Allocation complete\n";
         //[>
          //* DEBUG
          //int nodes_check = 0;
          //for (auto N = g->begin(); N != g->end(); ++N) {
            //++nodes_check;
            //std::cout << " -> " << nodes_check << "\n";
            ////Galois::Runtime::prefetch(*N,flag_none);
          //}
          //std::cout<<"Nodes_check = " << nodes_check << "\n";
          //*/

          //for (unsigned z = 0; z < num; ++z)
          //{
            //if(z != id)
            //{
              //std::cout <<" Z : " << z << "\n";
              //Runtime::getSystemNetworkInterface().sendAlt(z, &createOnHost<Args...>, g, inputFile, args...);
            //}
          //}

          //Runtime::getSystemBarrier();

          //// Add egdes for nodes on host 0
          //unsigned total_nodes = counts.size();
          //unsigned zero = 0;
          //unsigned start_index, end_index;
          //std::tie(start_index, end_index) = Galois::block_range(zero, total_nodes, id, num);

          ////g->prefetch_these(start_index, end_index);
//[>
          //auto p = block_range(counts.begin(), counts.end(), Runtime::NetworkInterface::ID, Runtime::NetworkInterface::Num);
          //unsigned start_index = std::distance(counts.begin(), p.first);
          //unsigned end_index = std::distance(p.first, p.second);
//*/

          
          //std::cout << " ON: " << Runtime::NetworkInterface::ID << " Nodes From [ " << start_index << " , " << end_index  << "]\n";
          //std::cout << "Adding outEdges and Inedges on : " << Runtime::NetworkInterface::ID << "\n";
          //for (unsigned x = start_index; x < end_index; ++x) {
            //auto fgn = *(fg.begin() + x);
            //auto gn = *(g->begin() + x);

            ////std::cout << gn.isLocal() << "\n";
            ////std::cout << " x : " << x << " gn : " << gn << " \n";
            //for (auto ii = fg.edge_begin(fgn), ee = fg.edge_end(fgn); ii != ee; ++ii) {
              //unsigned dst = fg.getEdgeDst(ii);
              //auto gn_dst = *(g->begin() + dst);
              ////std::cout << gn_dst.isLocal() << "\n";
              ////std::cout << " dst : " << dst << " gdst : " << (g->begin() + dst) << " \t";
              ////int val = fg.getEdgeData<int>(ii);
              ////g->addEdge(gn, *(g->begin() + dst), Galois::MethodFlag::SRC_ONLY);
              ////if((dst >= start_index) && (dst < end_index) ) //XXX: check this: Is local
                ////g->addInEdge(*(g->begin() + dst),gn, Galois::MethodFlag::SRC_ONLY);
            //}
          //}
          
        //}


#endif
