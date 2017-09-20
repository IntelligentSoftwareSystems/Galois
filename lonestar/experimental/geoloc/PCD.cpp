/*
 * Implementation of the algorithm proposed in the paper: Geotagging One Hundred Million Twitter Accounts with Total Variation Minimization
 *
 * References
 * [2] - Fritz, Heinrich, Peter Filzmoser, and Christophe Croux. 
 *       "A comparison of algorithms for the multivariate L 1-median." 
 *       Computational Statistics 27.3 (2012): 393-410.
 *
 * [2] - Compton, Ryan, David Jurgens, and David Allen. 
 * 	 "Geotagging one hundred million twitter accounts with total variation minimization." 
 * 	 Big Data (Big Data), 2014 IEEE International Conference on. IEEE, 2014.
 *
 * Date: August 12, 2015
 * Author: Altino Sampaio, FEUP, Portugal
 *
 */

#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Graphs/LCGraph.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>

static const bool DO_PRINT_MESSAGES	= false;

static const float MAX_VARIATION_GAMMA	= 50;	// the median value of neighbors location must be under this value so location can be inferred
static const int MAX_RANGE_LOCATION	= 10;	// coordinate values of nodes are generated under this value
static const int MAX_NUMBER_MENTIONS	= 5;	// it is the maximum value of an edge, which is determined as the number of reciprocated mentions

struct Location {
	float x;
	float y;
};

struct TNode {
	Location location;
	Location estimate;

	float geometric_median;
	char type;
	char state;
};

typedef galois::Graph::LC_CSR_Graph<TNode, int> Graph;
typedef Graph::GraphNode GNode;
static const unsigned int DIST_INFINITY = std::numeric_limits<unsigned int>::max();

namespace cll = llvm::cl;

static const char* name = "Parallel Coordinate Descent";
static const char* desc = "Computes the location of a Twitter user based in the kwnon location of their users";
static const char* url = "Paper: Geotagging One Hundred Million Twitter Accounts with Total Variation Minimization";

static cll::opt<std::string> filename( cll::Positional, cll::desc( "<input file>" ), cll::Required );
static cll::opt<unsigned int> maxIterations( "maxIterations", cll::desc( "Maximum iterations" ), cll::init( 10 ) );
static cll::opt<float> minUpdateLocation( "minUpdateLocation", cll::desc( "Minimum update of the location below which iterations stop" ), cll::init( 0.0 ) );
static cll::opt<unsigned int> probWithinSetU( "probWithinSetU", cll::desc( "Probability of creating nodes within set U (value between 0 and 100)" ), cll::init( 30 ) );
//static cll::opt<float> minUpdateLocation( "minUpdateLocation", cll::desc( "Minimum update for location to stop" ), cll::init( 0.15 ) );


Graph graph;

struct CountUNeighbors {
	Graph& g;
        galois::GAccumulator<int>& cnt_l_set_neighbors;

	CountUNeighbors( Graph& g, galois::GAccumulator<int>& cnt_l_set_neighbors ): g( g ), cnt_l_set_neighbors( cnt_l_set_neighbors ) { }
	void operator()( const GNode& n, galois::UserContext<GNode>& ctx ) const {
		TNode& node = g.getData( n );

		if ( node.type != 'U' ) return;

                for ( Graph::edge_iterator ii = g.edge_begin( n ), ei = g.edge_end( n ); ii != ei; ++ii ) {
                	GNode dst = g.getEdgeDst( ii );
                        TNode& neighbor = g.getData( dst );

			if ( neighbor.type == 'U' ) continue;
			cnt_l_set_neighbors += 1;
		}
	}
};

struct Initialize {
	Graph& g;

	Initialize( Graph& g ): g( g ) { }

	void operator()( const GNode& n, galois::UserContext<GNode>& ctx ) const {
		TNode& node = g.getData( n );

		if ( rand() % 100  > probWithinSetU ) {
			node.type = 'L';
			node.location.x = node.estimate.x = ( float ) ( rand() % MAX_RANGE_LOCATION );
			node.location.y = node.estimate.y = ( float ) ( rand() % MAX_RANGE_LOCATION );
		}
		else {
			node.type = 'U';
			node.location.x = node.estimate.x = 0.0;
			node.location.y = node.estimate.y = 0.0;
		}

		node.state = 'S';
		node.geometric_median = DIST_INFINITY;

    		for ( Graph::edge_iterator nn = g.edge_begin( n ), en = g.edge_end( n ); nn != en; ++nn ) {
      			g.getEdgeData( nn ) = rand() % MAX_NUMBER_MENTIONS;	// should be replaced by @mentions: see the paper
    		}

		if ( DO_PRINT_MESSAGES ) printf( "Type %c \t| x = %.2f \t| y = %.2f\n", node.type, node.location.x, node.location.y );
	}
};

struct ValueEqual {
	Graph& g;
  	char t;

  	ValueEqual( Graph& g, char t ): g( g ), t( t ) { }
	bool operator()( const GNode& n ) const {
    		return g.getData( n ).type == t;
  	}
};

struct FindTouched {
	Graph& g;

	FindTouched( Graph& g ): g( g ) { }
	bool operator()( const GNode& n ) const {
		TNode& node = g.getData( n );

		if ( node.type != 'U' ) return false;
                return node.state != 'S';
        }
};

float GetMedian( std::vector<float>& vector, int vector_size )
{
	// the following two loops sort the array x in ascending order
	for ( int i = 0; i < vector_size - 1; i++ ) {
		for ( int j = i + 1; j < vector_size; j++ ) {
			if ( vector[ j ] < vector[ i ] ) {
				// swap elements
				float temp = vector[ i ];
				vector[ i ] = vector[ j ];
				vector[ j ] = temp;
			}
		}
	}

	if( vector_size % 2 == 0 ) {
		// if there is an even number of elements, return mean of the two elements in the middle
		return ( ( vector[ vector_size / 2 ] + vector[ vector_size / 2 - 1 ] ) / 2.0 );
	} 
	else {
		// else return the element in the middle
		return vector[ vector_size / 2 ];
	}
}

struct EstimateLocation 
{
	Graph& g;
	galois::GAccumulator<int>& cnt_useful_it;
	galois::GAccumulator<int>& cnt_useless_it;
	galois::GAccumulator<float>& error_location;

	EstimateLocation( Graph& g, galois::GAccumulator<float>& error_location, galois::GAccumulator<int>& cnt_useful_it, galois::GAccumulator<int>& cnt_useless_it ): g( g ), error_location( error_location ), cnt_useful_it( cnt_useful_it ), cnt_useless_it( cnt_useless_it ) { }

	void operator()( GNode& n, galois::UserContext<GNode>& ctx ) const {
		TNode& node = g.getData( n );

		if ( node.type == 'U' and node.state != 'F' ) {
			size_t num_neighbors = std::distance( g.edge_begin( n ), g.edge_end( n ) );

			Location sum_num;               // sum for the numerator
			float sum_den = 0.0;            // sum for the denominator
			float geometric_median = 0.0;
 
			short eta = 0;                  // by default, we assume no colisions, meaning that the solution will converge
			Location R;                     // Equation 9 in [1]
			Location T;                     // Equation 7 in [1]
			float gamma = 0.0;              // Equation 10 in [1]

			std::vector<float> neighbor_list( num_neighbors );
			bool valid_neighbors = false;
			int cnt_neighbors = 0;

			sum_num.x = 0.0;
			sum_num.y = 0.0;
	
			R.x = 0.0;
			R.y = 0.0;	

			for ( Graph::edge_iterator ii = g.edge_begin( n ), ei = g.edge_end( n ); ii != ei; ++ii ) {
				GNode dst = g.getEdgeDst( ii );
				TNode& neighbor = g.getData( dst );
				int weight = g.getEdgeData( ii );

				if ( neighbor.type != 'L' ) continue;

if ( DO_PRINT_MESSAGES ) printf( "Node type = %c | Edge_w = %d\n", node.type, weight );

				// median
				neighbor_list[ cnt_neighbors++ ] = sqrt( ( neighbor.location.x - node.location.x ) * ( neighbor.location.x - node.location.x ) + 
									( neighbor.location.y - node.location.y ) * ( neighbor.location.y - node.location.y ) );

				if ( neighbor.location.x != node.location.x || neighbor.location.y != node.location.y ) {
					float euclidean_distance = sqrt( ( neighbor.location.x - node.location.x ) * ( neighbor.location.x - node.location.x ) + 
									( neighbor.location.y - node.location.y ) * ( neighbor.location.y - node.location.y ) );
if ( euclidean_distance == 0.0 ) {
	printf( "Bad mistake...0 \n" );
	exit( 0 );
}
					// {1} determine Euclidean distance
					geometric_median += weight * euclidean_distance;	// equation 6 in [2]
					// {1} end

					// {2} estimate new location
					sum_num.x += neighbor.location.x / euclidean_distance;	// x
					sum_num.y += neighbor.location.y / euclidean_distance;	// y
					sum_den += 1 / euclidean_distance;

					R.x += ( neighbor.location.x - node.location.x ) / euclidean_distance;	// x
					R.y += ( neighbor.location.y - node.location.y ) / euclidean_distance;	// y
					// {2} end

					valid_neighbors = true;
				}
				else {
					eta = 1;
				}
			}

			if ( cnt_neighbors <= 1 || valid_neighbors == false ) return;	// == 0

if ( sum_den == 0 ) { 
	printf( "Bad mistake...1 \n" );
	exit( 0 );
}

			T.x = sum_num.x / sum_den;
			T.y = sum_num.y / sum_den;

			float euclidean_distance_R = sqrt( R.x * R.x + R.y * R.y );

			if ( euclidean_distance_R == 0.0 ) {
				gamma = 1;
			}
			else {
				gamma = ( ( 1 > eta / euclidean_distance_R ) ? eta / euclidean_distance_R : 1 );
			}

			float median = GetMedian( neighbor_list, cnt_neighbors );
if ( DO_PRINT_MESSAGES ) printf( "Median = %.4f\n", median );
			if ( median >= MAX_VARIATION_GAMMA ) return;

if ( DO_PRINT_MESSAGES ) printf( "Geometric median | old = %.4f | new = %.4f\n", node.geometric_median, geometric_median );

			float diff_loca = node.geometric_median - geometric_median;
			float estimated_x = ( 1 - gamma ) * T.x + gamma * node.location.x;        // Equation 11 in [1]
                        float estimated_y = ( 1 - gamma ) * T.y + gamma * node.location.y;        // Equation 11 in [1]
                        float update_loca = sqrt( ( node.location.x - estimated_x ) * ( node.location.x - estimated_x ) +
                                                ( node.location.y - estimated_y ) * ( node.location.y - estimated_y ) );

                        if ( node.state == 'S' ) node.state = 'P';
                        if ( update_loca < minUpdateLocation ) node.state = 'F';

                        node.location.x = estimated_x;
                        node.location.y = estimated_y;

                        if ( node.geometric_median > geometric_median ) {
                                node.geometric_median = geometric_median;
                                node.estimate.x = estimated_x;
                                node.estimate.y = estimated_y;

				cnt_useful_it += 1;

if ( DO_PRINT_MESSAGES ) printf( "Estimate.x = %.4f, Estimate.y = %.4f\n", node.estimate.x, node.estimate.y );
                        }
			else {
				error_location += abs( diff_loca );
				cnt_useless_it += 1;
if ( DO_PRINT_MESSAGES ) printf( "\t>>>> Error = %.4f\n", diff_loca );
			}

//if ( DO_PRINT_MESSAGES ) printf( "Location.x = %.4f, Location.y = %.4f, update_amplitude %.4f\n", node.location.x, node.location.y, diff_loca ); //update_loca );

// ----------------------

		}
	}

};

int main( int argc, char** argv ) 
{
	galois::StatManager statManager;
	LonestarStart( argc, argv, name, desc, 0 );

	galois::Timer T;

	srand( time( NULL ) );
	galois::GAccumulator<int> cnt_useful_it;
	galois::GAccumulator<int> cnt_useless_it;
	galois::GAccumulator<int> cnt_l_set_neighbors;
	galois::GAccumulator<float> error_location;
	galois::Graph::readGraph( graph, filename );
	galois::for_each_local( graph, Initialize( graph ) );

        int u_set_size = std::count_if( graph.begin(), graph.end(), ValueEqual( graph, 'U' ) );
        int l_set_size = std::count_if( graph.begin(), graph.end(), ValueEqual( graph, 'L' ) );

	T.start();
	for ( int iteration = 0; iteration < maxIterations; ++iteration ) {
		// Unlike galois::for_each, galois::for_each_local initially assigns work
  		// based on which thread created each node (galois::for_each uses a simple
  		// blocking of the iterator range to initialize work, but the iterator order
  		// of a Graph is implementation-defined). 
  		galois::for_each_local( graph, EstimateLocation( graph, error_location, cnt_useful_it, cnt_useless_it ) );
		//galois::for_each( graph.begin(), graph.end(), EstimateLocation( graph, error_location, cnt_useful_it, cnt_useless_it ) );
	}
	T.stop();

std::cout << "--------------------------------------------------------------------------------------------------------------------\n";
	std::cout << "Notes: "
		    	"\tusers which location is know belong to set L\n"
			"\tour goal is to assign a location to nodes in U\n"
			"\tthe vertex of our social network is thus partitioned as: V = L + U\n\n";
	std::cout << "Elapsed time: " << T.get() << " milliseconds\n";

	int u_set_touched = std::count_if( graph.begin(), graph.end(), FindTouched( graph ) );
	galois::for_each( graph.begin(), graph.end(), CountUNeighbors( graph, cnt_l_set_neighbors ) );

	std::cout << "Number of nodes in U: " << u_set_size << "\n";
	std::cout << "Number of nodes in L: " << l_set_size << "\n";
	std::cout << "Number of nodes in V: " << graph.size() << "\n";

	float avg_numb_iterations = ( u_set_touched > 0 ? ( float ) cnt_useful_it.reduce() / ( float ) u_set_touched : -1 );
	float avg_error_location = ( ( float ) cnt_useless_it.reduce() > 0.0 ? error_location.reduce() / ( float ) cnt_useless_it.reduce() : 0 );

//printf( "2: %.3f\n", ( float ) error_location.reduce() );
//printf( "3: %d\n", cnt_useless_it.reduce() );

	printf( "Average number of L neighbors per U node: %.3f\n", ( float ) cnt_l_set_neighbors.reduce() / ( float ) u_set_size );
	printf( "Average error related to the best location: %.3f\n", avg_error_location );
	printf( "Average number of iterations to achieve the best location (i.e., minimum geometric median): %.3f\n", avg_numb_iterations );
std::cout << "--------------------------------------------------------------------------------------------------------------------\n";

	return 0;
}
