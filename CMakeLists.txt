cmake_minimum_required(VERSION 3.10)
project(cudaRandomForest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    bagging.cpp
    dataReader.cpp
    decisiontreeClassifier.cpp
    metrics.cpp
    # metrics.cu
    # normalization.cpp
    randomForestClassifier.cpp
)

set(HEADERS
    bagging.hpp
    dataReader.hpp
    decisiontreeClassifier.hpp
    # metrics.cuh
    metrics.hpp
    randomForestClassifier.hpp
)

add_executable(cudaRandomForest ${SOURCES} ${HEADERS})