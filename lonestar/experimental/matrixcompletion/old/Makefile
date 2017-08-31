
INCLUDES = -I$(BOOST_INC) -I./../../../../build/release/include -I./../../../../include
LINKS = -I$(BOOST_LIB) -L./../../../../build/release/src -lgalois -pthread /opt/intel/vtune_amplifier_xe_2013/lib64/libittnotify.a -ldl 

all:
	g++ -g -O3 -ffast-math -std=c++11 sgd.cpp -o sgd $(INCLUDES) $(LINKS)
