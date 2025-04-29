# cython: language_level=3
# distutils: language = c++
# distutils: sources = ../randomForestClassifier.cpp

from libcpp cimport bool
from libc.stdint cimport uintptr_t
import numpy as np
cimport numpy as cnp

cdef extern from "../randomForestClassifier.hpp":
    cdef cppclass randomForestClassifier:
        randomForestClassifier() except +
        randomForestClassifier(unsigned class_count, 
                             unsigned min_samples_split, 
                             unsigned n_trees) except +
        
        void fit(float* features, unsigned* labels) except +
        void fit(float* features, unsigned* labels, 
                unsigned sample_count_, 
                unsigned feature_count_, 
                unsigned class_count_) except +
        
        void predict(float* features, unsigned* labels, 
                    unsigned sample_count) except +

# Python包装类
cdef class PyRandomForestClassifier:
  cdef randomForestClassifier* thisptr  # 指向C++对象的指针
  
  def __cinit__(self, unsigned class_count=2, 
                unsigned min_samples_split=5, 
                unsigned n_trees=100):
    self.thisptr = new randomForestClassifier(
        class_count, min_samples_split, n_trees)
  
  def __dealloc__(self):
    del self.thisptr
  
  def fit(self, X, y):
    cdef cnp.ndarray[cnp.float32_t, ndim=2] X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float32), dtype=np.float32)
    
    if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError("X不能为空数组")
    
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] y_arr = np.ascontiguousarray(np.asarray(y, dtype=np.uint32), dtype=np.uint32)
    
    if y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("X和y的样本数量不匹配")
    
    cdef float[:, ::1] X_view = X_arr
    cdef unsigned[::1] y_view = y_arr
    
    self.thisptr.fit(&X_view[0][0], &y_view[0],
                    X_arr.shape[0], X_arr.shape[1],
                    int(np.max(y_arr)) + 1)

  def predict(self, X):
    cdef cnp.ndarray[cnp.float32_t, ndim=2] X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float32), dtype=np.float32)
    if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError("X不能为空数组")
    
    cdef float[:, ::1] X_view = X_arr
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] y_pred = np.empty(
        X_arr.shape[0], dtype=np.uint32)
    cdef unsigned[::1] y_pred_view = y_pred
    
    self.thisptr.predict(&X_view[0][0], &y_pred_view[0], X_arr.shape[0])
    
    return y_pred
