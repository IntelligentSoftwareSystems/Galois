#/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

echo "Installing/updating essential tools..."

case "$(uname)" in
  Darwin)
    echo "OSX is not yet supported"
    exit 0
    ;;
  Linux*)
    sudo apt update
    sudo apt install -y git python3 python3-pip build-essential bc cmake
    ;;
esac

echo "Cloning utcs-scea/ava..."
cd $SCRIPTPATH/..
git clone https://github.com/utcs-scea/ava.git
cd $SCRIPTPATH/../ava

echo "Cloning ava/llvm..."
git checkout 6cfc3c8e5c58dfba
git submodule init
git submodule update --recursive

echo "Building ava/llvm..."
cd $SCRIPTPATH/../ava/llvm
mkdir build >/dev/null 2>&1
cd build
cmake -DLLVM_ENABLE_PROJECTS=clang -G "Unix Makefiles" ../llvm
make -j4

echo "Installing Python packages..."
cd $SCRIPTPATH/../ava/cava
python3 -m pip install --user --requirement requirements.txt

echo "Install tool chains..."
case "$(uname)" in
  Darwin)
    brew install llvm@7
    ;;
  Linux*)
    sudo apt install -y libclang-7-dev clang-7 indent libglib2.0-dev
    ;;
esac

echo "Generating API remoting stubs (test)..."
cd $SCRIPTPATH/..
./ava/cava/nwcc specs/cuda_10_1.c -I /usr/local/cuda-10.1/include -I ava/cava/headers `pkg-config --cflags glib-2.0`
