from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "random_forest",
    sources=["wrapper.pyx", "../randomForestClassifier.cpp",  "../decisiontreeClassifier.cpp",  "../bagging.cpp", 
    "../dataReader.cpp",  
    "../metrics.cpp",
    "../randomForestRegressor.cpp"],
    language="c++",
    include_dirs=[np.get_include()],  # 包含numpy头文件
    extra_compile_args=["-std=c++11"],  # 根据需要添加编译选项
)

setup(
    name="random_forest",
    version="1.0",
    description="Python wrapper for RandomForest",
    ext_modules=cythonize(ext),
)