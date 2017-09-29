#ifndef TCPSERVER_H
#define TCPSERVER_H

#include "MC.h"
#include "galois/Endian.h"
#include "galois/ParallelSTL/ParallelSTL.h"
#include "galois/runtime/TiledExecutor.h"

#include <boost/asio.hpp>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

template<typename Graph>
struct AuxData {
  typedef typename Graph::GraphNode GNode;

  Graph& graph;
  size_t numItems;
  std::vector<std::pair<double, size_t> > margins;
  std::ostream& out;
  std::map<int, size_t> itemCount;
  std::map<int, size_t> userCount;
  size_t numDeletedItems;
  size_t numDeletedUsers;
  
  AuxData(Graph& g, size_t numItems, std::ostream& out):
    graph(g), numItems(numItems), out(out), numDeletedItems(0), numDeletedUsers(0) { }

  void initialize() {
    updateMargins();
    galois::runtime::Fixed2DGraphTiledExecutor<Graph> executor(graph);
    executor.execute(
        graph.begin(), graph.begin() + numItems,
        graph.begin() + numItems, graph.end(),
        1000, 1000,
        [&](GNode src, GNode dst, typename Graph::edge_iterator) {
          graph.getData(src).count += 1;
          graph.getData(dst).count += 1;
        }, true);
  }

  void update() {
    updateMargins();
    //TODO updateDeleted();
  }

  void updateMargins() {
    typedef typename Graph::GraphNode GNode;
    galois::Timer elapsed;

    elapsed.start();
    galois::runtime::Fixed2DGraphTiledExecutor<Graph> executor(graph);
    double norm1 = 1.0 / (graph.size() - numItems);
    double norm2 = 1.0 / (numItems);
    executor.executeDense(
        graph.begin(), graph.begin() + numItems,
        graph.begin() + numItems, graph.end(),
        100, 100,
        [&](GNode src, GNode dst) {
          auto e = predictionError(
              graph.getData(src).latentVector,
              graph.getData(dst).latentVector,
              0);
          graph.getData(src).sum += e;
          graph.getData(dst).sum += e;
        }, true);
    galois::do_all(graph, [&](GNode n) {
      if (n < numItems)
        graph.getData(n).sum *= norm1;
      else
        graph.getData(n).sum *= norm2;
    });
    elapsed.stop();
    double flop = (double) (graph.size() - numItems) * numItems * (2.0 * 100 + 2);
    flop += graph.size();

    //std::cout << "Margin GFLOP/S: " << flop / elapsed.get() / 1e6 << "\n"; // XXX
    margins.resize(graph.size());
    auto mm = margins.begin();
    size_t id = 0;
    for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii, ++mm, ++id) {
      *mm = std::make_pair(graph.getData(*ii).sum, id);
    }
    galois::ParallelSTL::sort(margins.begin(), margins.begin() + numItems);
    galois::ParallelSTL::sort(margins.begin() + numItems, margins.end());
  }
};

template<typename Algo, typename Graph>
class TcpConnection: public std::enable_shared_from_this<TcpConnection<Algo, Graph> > {
  typedef typename Graph::GraphNode GNode;

  boost::asio::ip::tcp::socket socket_;
  Algo& algo;
  Graph& graph;
  AuxData<Graph>& auxData;
  std::ostream& out;

  galois::runtime::Fixed2DGraphTiledExecutor<Graph> executor;
  boost::asio::streambuf request;
  galois::runtime::PerThreadStorage<std::stringstream> response;
  std::stringstream sumResponse;
    
  std::string kind;
  size_t uidBegin;
  size_t uidEnd;
  size_t iidBegin;
  size_t iidEnd;
  int minValue;
  int maxValue;

  void handleRequest(const boost::system::error_code& error, size_t) {
    if (error) {
      out << "Error in reading request " << error << "\n";
      return;
    }

    std::istream in(&request);
    in >> kind >> uidBegin >> uidEnd >> iidBegin >> iidEnd >> minValue >> maxValue;
    if (!in) {
      out << "Error parsing request\n";
    } else if (kind == "GET") {
      handleGetRequest();
    } else if (kind == "GET_PRED") {
      handleGetPredictionRequest();
    } else if (kind == "SUMMARY") {
      handleSummaryRequest();
    } else if (kind == "DELETE") {
      handleDeleteRequest();
    } else if (kind == "MODIFY") {
      handleModifyRequest();
    } else if (kind == "TOP") {
      handleTopRequest();
    }

    for (unsigned i = 0; i < response.size(); ++i) {
      sumResponse << response.getRemote(i)->str();
    }

    boost::asio::async_write(
        socket_,
        boost::asio::buffer(sumResponse.str()),
        std::bind(
          &TcpConnection::handleWrite,
          this->shared_from_this(),
          std::placeholders::_1,
          std::placeholders::_2));
  }

  bool validRange() {
    size_t numItems = algo.numItems();
    size_t r = std::distance(graph.begin(), graph.end());
    if (uidBegin >= r || uidEnd > r || iidBegin >= r || iidEnd > r)
      return false;
    if (uidBegin < numItems || uidEnd < numItems)
      return false;
    if (iidBegin >= numItems || iidEnd > numItems)
      return false;
    return true;
  }

  void recompute() {
    initializeGraphData(graph); // TODO reinitialize only part of latent space
    std::unique_ptr<StepFunction> sf { newStepFunction() };
    algo(graph, *sf);
    auxData.update();
  }

  void writeNetworkInt(std::ostream& o, uint32_t v) {
    v = galois::convert_htobe32(v);
    o.write(reinterpret_cast<char*>(&v), sizeof(v));
  }

  void writeNetworkFloat(std::ostream& o, float v) {
    union { float as_float; uint32_t as_uint; } c = { v };
    uint32_t value = galois::convert_htobe32(c.as_uint);
    o.write(reinterpret_cast<char*>(&value), sizeof(value));
  }

  void handleSummaryRequest() {
    writeNetworkInt(*response.getLocal(), auxData.numItems);
    writeNetworkInt(*response.getLocal(), auxData.numDeletedItems);
    writeNetworkInt(*response.getLocal(), graph.size() - auxData.numItems);
    writeNetworkInt(*response.getLocal(), auxData.numDeletedUsers);
  }

  void handleGetRequest() {
    if (!validRange())
      return;
    executor.execute(
        graph.begin() + iidBegin, graph.begin() + iidEnd,
        graph.begin() + uidBegin, graph.begin() + uidEnd,
        1000, 1000,
        [&](GNode src, GNode dst, typename Graph::edge_iterator edge) {
          writeNetworkInt(*response.getLocal(), src);
          writeNetworkInt(*response.getLocal(), graph.getData(src).count);
          writeNetworkInt(*response.getLocal(), dst);
          writeNetworkInt(*response.getLocal(), graph.getData(dst).count);
          writeNetworkFloat(*response.getLocal(), graph.getEdgeData(edge));
        }, false);
  }

  void handleModifyRequest() {
    if (!validRange())
      return;
    if (minValue != maxValue)
      return;

    executor.execute(
        graph.begin() + iidBegin, graph.begin() + iidEnd,
        graph.begin() + uidBegin, graph.begin() + uidEnd,
        1000, 1000,
        [&](GNode src, GNode dst, typename Graph::edge_iterator edge) {
          graph.getEdgeData(edge) = minValue;
        }, false);
    recompute(); // XXX
  }

  void handleDeleteRequest() {
    if (!validRange())
      return;
    for (auto ii = graph.begin() + iidBegin, ei = graph.begin() + iidEnd; ii != ei; ++ii)
      graph.getData(*ii).deleted = true;
    for (auto ii = graph.begin() + uidBegin, ei = graph.begin() + uidEnd; ii != ei; ++ii)
      graph.getData(*ii).deleted = true;
    recompute(); // XXX
  }

  void handleTopRequest() {
    size_t numItems = algo.numItems();
    // Either users or items but not both
    if ((uidBegin == uidEnd) == (iidBegin == iidEnd))
      return;
    if ((minValue > 0) == (maxValue > 0))
      return;
    int count = 0;
    if (minValue > 0) {
      size_t first = (uidBegin != uidEnd) ? numItems : 0; 
      size_t last = (uidBegin != uidEnd) ? auxData.margins.size() : numItems;
      for (auto ii = auxData.margins.begin() + first, ei = auxData.margins.begin() + last; ii != ei && count < minValue; ++ii) {
        if (!graph.getData(graph.begin()[ii->second]).deleted) {
          writeNetworkInt(*response.getLocal(), ii->second);
          writeNetworkInt(*response.getLocal(), graph.getData(ii->second).count);
          writeNetworkFloat(*response.getLocal(), ii->first);
          count += 1;
        }
      }
    } else {
      size_t first = (uidBegin != uidEnd) ? 0 : auxData.margins.size() - numItems; 
      size_t last = (uidBegin != uidEnd) ? numItems : auxData.margins.size();
      for (auto ii = auxData.margins.rbegin() + first, ei = auxData.margins.rbegin() + last; ii != ei && count < maxValue; ++ii) {
        if (!graph.getData(graph.begin()[ii->second]).deleted) {
          writeNetworkInt(*response.getLocal(), ii->second);
          writeNetworkInt(*response.getLocal(), graph.getData(ii->second).count);
          writeNetworkFloat(*response.getLocal(), ii->first);
          count += 1;
        }
      }
    }
  }

  void handleGetPredictionRequest() {
    if (!validRange())
      return;
    executor.executeDense(
        graph.begin() + iidBegin, graph.begin() + iidEnd,
        graph.begin() + uidBegin, graph.begin() + uidEnd,
        100, 100,
        [&](GNode src, GNode dst) {
          auto e = predictionError(
              graph.getData(src).latentVector,
              graph.getData(dst).latentVector,
              0);
          if (minValue && e < minValue)
            return;
          if (maxValue && e >= maxValue)
            return;
          writeNetworkInt(*response.getLocal(), src);
          writeNetworkInt(*response.getLocal(), graph.getData(src).count);
          writeNetworkInt(*response.getLocal(), dst);
          writeNetworkInt(*response.getLocal(), graph.getData(dst).count);
          writeNetworkFloat(*response.getLocal(), (float) e);
        }, false);
  }

  void handleWrite(const boost::system::error_code& error, size_t) {
    if (error) {
      out << "Error sending response " << error << "\n";
    }
  }

public:
  TcpConnection(
      boost::asio::io_service& io_service, 
      Algo& a,
      Graph& g,
      AuxData<Graph>& auxData,
      std::ostream& o):
    socket_(io_service), algo(a), graph(g), auxData(auxData), out(o), executor(g) { }

  boost::asio::ip::tcp::socket& socket() {
    return socket_;
  }

  void start() {
    boost::asio::async_read_until(
        socket_,
        request,
        "\r\n",
        std::bind(
          &TcpConnection::handleRequest,
          this->shared_from_this(),
          std::placeholders::_1,
          std::placeholders::_2));
  }
};

template<typename Algo, typename Graph>
class TcpServer {
  typedef TcpConnection<Algo, Graph> Connection;
  typedef std::shared_ptr<Connection> tpointer;
  boost::asio::ip::tcp::acceptor acceptor;
  Graph& graph;
  Algo& algo;
  AuxData<Graph>& auxData;
  std::ostream& out;

  void startAccept() {
    tpointer con = tpointer(new Connection(acceptor.get_io_service(), algo, graph, auxData, out));
    acceptor.async_accept(
        con->socket(),
        std::bind(&TcpServer::handleAccept, this, con, std::placeholders::_1));
  }

  void handleAccept(tpointer con, const boost::system::error_code& error) {
    if (!error) {
      con->start();
    }
    startAccept();
  }

public:
  TcpServer(
      boost::asio::io_service& io_service, 
      int port,
      Algo& a,
      Graph& g,
      AuxData<Graph>& auxData,
      std::ostream& o):
    acceptor(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port)),
    graph(g), algo(a), auxData(auxData), out(o)
  {
    startAccept();
  }
};

template<typename Algo, typename Graph>
auto startServerHelper(Algo& algo, Graph& graph, int port, std::ostream& out, int) 
  -> decltype(typename Graph::node_data_type().sum, bool()) 
{
  AuxData<Graph> auxData(graph, algo.numItems(), out);
  auxData.initialize();

  try {
    boost::asio::io_service io_service;
    TcpServer<Algo, Graph> server(io_service, port, algo, graph, auxData, out);
    out << "Server up!\n";
    io_service.run();
  } catch (std::exception& e) {
    out << e.what() << "\n";
  }
  return false;
}

template<typename Algo, typename Graph>
void startServerHelper(Algo& algo, Graph& graph, int port, std::ostream& out, ...) {
  GALOIS_DIE("server mode not supported for this graph type");
}

template<typename Algo, typename Graph>
void startServer(Algo& algo, Graph& graph, int port, std::ostream& out) {
  startServerHelper(algo, graph, port, out, 0);
}
#endif
