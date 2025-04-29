import os, numba.cuda, cupy

print("CUDA Paths:")
print(f"CUDA_HOME: {os.getenv('CUDA_HOME')}")
print(f"LD_LIBRARY_PATH: {os.getenv('LD_LIBRARY_PATH')}")

print("\nDevice Check:")
print(f"Numba devices: {numba.cuda.gpus}")
print(f"CuPy device: {cupy.cuda.runtime.getDeviceCount()} devices")