rm -rf build *.so *.cpp
python setup.py build_ext --inplace --force
mv random_forest.cpython-312-x86_64-linux-gnu.so ../python/
python3 ../python/RandomForest_Mine.py