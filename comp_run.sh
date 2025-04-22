#!/bin/bash

mkdir -p ./build
cd build

# 检查是否已经运行过cmake
if [ ! -f "Makefile" ]; then
  cmake ..
fi

cmake --build .

# 返回父目录并运行程序
cd ..

./build/cudaRandomForest