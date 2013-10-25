
INCLUDES = -I$(BOOST_INC) -I./Galois-2.1.8/build/release/include -I./Galois-2.1.8/include
LINKS = -I$(BOOST_LIB) -L./Galois-2.1.8/build/release/src -lgalois -pthread /opt/intel/vtune_amplifier_xe_2013/lib64/libittnotify.a -ldl 

all:
	g++ -g -O3 -ffast-math -std=c++11 sgd.cpp -o sgd $(INCLUDES) $(LINKS)
