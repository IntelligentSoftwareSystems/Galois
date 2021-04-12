# This app has to be built with -O1 to avoid a compiler bug.
set(CMAKE_CUDA_FLAGS_RELEASE "-O1 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O1 -g -DNDEBUG")
app_scientific_gpu(bh barneshut)
add_test(NAME barneshut COMMAND barneshut-gpu 50000 2 0)
