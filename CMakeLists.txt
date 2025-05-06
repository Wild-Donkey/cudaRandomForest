cmake_minimum_required(VERSION 3.10)
project(cudaRandomForest LANGUAGES CXX)
# add_compile_options(-g -Wall -O3)
add_compile_options(-g -Wall -O3 -fsanitize=address -fno-omit-frame-pointer)
add_link_options(-fsanitize=address -fno-omit-frame-pointer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    bagging.cpp
    dataReader.cpp
    # decisiontreeClassifier.cpp
    decisiontreeRegressor.cpp
    metrics.cpp
    # metrics.cu
    # normalization.cpp
    # randomForestClassifier.cpp
    randomForestRegressor.cpp
)

set(HEADERS
    bagging.hpp
    dataReader.hpp
    # decisiontreeClassifier.hpp
    decisiontreeRegressor.hpp
    # metrics.cuh
    metrics.hpp
    # randomForestClassifier.hpp
    randomForestRegressor.hpp
)

add_executable(cudaRandomForest ${SOURCES} ${HEADERS})