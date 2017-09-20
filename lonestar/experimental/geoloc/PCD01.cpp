#include "galois/Galois.h"
#include "galois/Accumulator.h"
#include "galois/Timer.h"
#include "galois/Graph/LCGraph.h"
#include "Lonestar/BoilerPlate.h"
#include "llvm/Support/Allocator.h"

static const int NUMBER_OF_ITERATIONS	= 10;	// maximum number of iterations to try convergence
static const int PROBABILITY_WITHIN_U	= 30;	// probability of a node to be within set U, (set U contains nodes which location is unknown)
static const float MAX_VARIATION_GAMMA	= 50;	// the median value of neighbors location must be under this value so location can be infered
static const int MAX_RANGE_LOCATION	= 10;	// coordinate values of nodes are generated under this value
static const int MAX_NUMBER_MENTIONS	= 5;	// it is the maximum value of an edge, which is determined as the number of reciprocated mentions
static const float MIN_UPDATE_LOCATION	= 0.25;	// stop iterative location update if location update is under such value

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

typedef galois::graphs::LC_CSR_Graph<TNode, int> Graph;
typedef Graph::GraphNode GNode;
static const unsigned int DIST_INFINITY = std::numeric_limits<unsigned int>::max();

//namespace mem = llvm::MallocAllocator;
namespace cll = llvm::cl;

static cll::opt<std::string> filename( cll::Positional, cll::desc( "<input file>" ), cll::Required );


Graph graph;

struct Initialize {
	galois::GAccumulator<int>& accum_set_size;
	Initialize( galois::GAccumulator<int>& accum_set_size ): accum_set_size( accum_set_size ) { }

	void operator()( const GNode& n, galois::UserContext<GNode>& ctx ) const {
		TNode& node = graph.getData( n );

		if ( rand() % 100  > PROBABILITY_WITHIN_U ) {
			node.type = 'L';
			node.location.x = node.estimate.x = ( float ) ( rand() % MAX_RANGE_LOCATION );
			node.location.y = node.estimate.y = ( float ) ( rand() % MAX_RANGE_LOCATION );
		}
		else {
			node.type = 'U';
			node.location.x = node.estimate.x = 0.0;
			node.location.y = node.estimate.y = 0.0;

			accum_set_size += 1;
		}

		node.state = 'S';
		node.geometric_median = DIST_INFINITY;

    		for ( Graph::edge_iterator nn = graph.edge_begin( n ), en = graph.edge_end( n ); nn != en; ++nn ) {
      			graph.getEdgeData( nn ) = rand() % MAX_NUMBER_MENTIONS;
    		}

		printf( "Type %c \t| x = %.2f \t| y = %.2f \t| U_set_size %d\n", node.type, node.location.x, node.location.y, accum_set_size.reduce() );
	}
};

float GetMedian( float* vector, int vector_size )
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

/*
for ( int i = 0; i < vector_size; i++ ) {
	printf( "> vector[%d] = %.2f\n", i, vector[ i ] );
}
printf( "\n" );
*/

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
	galois::GAccumulator<int>& accum_set_size;
	EstimateLocation( Graph& g, galois::GAccumulator<int>& accum_set_size ): g( g ), accum_set_size( accum_set_size ) { }

	void operator()( const GNode& n, galois::UserContext<GNode>& ctx ) const {
		TNode& node = g.getData( n );

		Location sum_num;		// sum for the numerator
		float sum_den = 0.0;		// sum for the denominator
		float geometric_median = 0.0;

		short eta = 0;			// by default, we assume no colisions, meaning that the solution will converge
		Location R;			// Equation 9 in [1]
		Location T;			// Equation 7 in [1]
		float gamma = 0.0;		// Equation 10 in [1]

		if ( node.type == 'U' and node.state != 'F' ) {
			size_t x = 5;
			llvm::MallocAllocator memory_space;
			float* vector = memory_space.Allocate<float>( ( size_t ) accum_set_size.reduce() );

			int vector_position = 0;

			for ( Graph::edge_iterator ii = g.edge_begin( n ), ei = g.edge_end( n ); ii != ei; ++ii ) {
				GNode dst = g.getEdgeDst( ii );
				TNode& neighbor = g.getData( dst );
				int weight = g.getEdgeData( ii );

				if ( neighbor.type != 'L' ) continue;

printf( "Node type = %c | Edge_w = %d\n", node.type, weight );

				// median
				vector[ vector_position++ ] = sqrt( ( neighbor.location.x - node.location.x ) * ( neighbor.location.x - node.location.x ) + 
								( neighbor.location.y - node.location.y ) * ( neighbor.location.y - node.location.y ) );

				if ( neighbor.location.x != node.location.x || neighbor.location.y != node.location.y ) {
					float euclidean_distance = sqrt( ( neighbor.location.x - node.location.x ) * ( neighbor.location.x - node.location.x ) + 
									( neighbor.location.y - node.location.y ) * ( neighbor.location.y - node.location.y ) );

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
				}
				else {
					eta = 1;
				}
			}

			if ( vector_position == 0 ) return;

			T.x = sum_num.x / sum_den;
			T.y = sum_num.y / sum_den;

			float euclidean_distance_R = sqrt( R.x * R.x + R.y * R.y );
			gamma = ( ( 1 > eta / euclidean_distance_R ) ? eta / euclidean_distance_R : 1 );

			float median = GetMedian( vector, vector_position );
printf( "Median = %.2f\n", median );
			if ( median >= MAX_VARIATION_GAMMA ) return;

printf( "Geometric median | old = %.2f | new = %.2f\n", node.geometric_median, geometric_median );

			if ( node.geometric_median > geometric_median ) {
				float estimated_x = ( 1 - gamma ) * T.x + gamma * node.location.x;        // Equation 11 in [1]
				float estimated_y = ( 1 - gamma ) * T.y + gamma * node.location.y;        // Equation 11 in [1]
				float update_loca = sqrt( ( node.location.x - estimated_x ) * ( node.location.x - estimated_x ) + 
							( node.location.y - estimated_y ) * ( node.location.y - estimated_y ) );

				if ( node.state == 'S' ) node.state = 'P';
				if ( update_loca < MIN_UPDATE_LOCATION ) node.state = 'F';

				node.geometric_median = geometric_median;
				node.location.x = estimated_x;
				node.location.y = estimated_y;

printf( "Location.x = %.2f, Location.y = %.2f, update_amplitude %.2f\n", node.location.x, node.location.y, update_loca );
			}

			memory_space.Deallocate( vector );
		}
	}

};

int main( int argc, char** argv ) 
{
	galois::StatManager statManager;
  	LonestarStart( argc, argv, 0, 0, 0 );
	galois::Timer T;

	srand( time( NULL ) );
	galois::GAccumulator<int> accum_set_size;
	galois::graphs::readGraph( graph, filename );
	galois::for_each( graph.begin(), graph.end(), Initialize( accum_set_size ) );

	T.start();
	for ( int iteration = 0; iteration < NUMBER_OF_ITERATIONS; ++iteration ) {
		galois::for_each( graph.begin(), graph.end(), EstimateLocation( graph, accum_set_size ) );
	}
	T.stop();

	std::cout << "Elapsed time: " << T.get() << " milliseconds\n";

	return 0;
}
