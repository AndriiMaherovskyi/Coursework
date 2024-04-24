#import pycuda.driver as cuda
import numpy as np
from pycuda import driver, compiler, gpuarray
import pycuda.autoinit  # Автоматичне ініціалізування контексту

driver.init()

# Create a CUDA context
device = driver.Device(0) # Визначає кількість підключених відеокарт, рахуємо від 0
context = device.make_context()

# Define the CUDA kernel
kernel_code = """
__global__ void add_arrays(float *a, float *b, float *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
"""

# Compile the CUDA kernel
module = compiler.SourceModule(kernel_code)

'''
    Поки залишаю приклад з сайту по масивах, потім будемо щось тут створювати
'''
# Allocate memory on the GPU
a_gpu = gpuarray.to_gpu(np.random.randn(100).astype(np.float32))
b_gpu = gpuarray.to_gpu(np.random.randn(100).astype(np.float32))
c_gpu = gpuarray.empty_like(a_gpu)

# Create events for timing
start = driver.Event()
end = driver.Event()

# Start timing
start.record()

# Launch the CUDA kernel
add_arrays = module.get_function("add_arrays")
add_arrays(a_gpu, b_gpu, c_gpu, block=(100,1,1))

# Stop timing
end.record()
end.synchronize()  # Очікування завершення всіх операцій
elapsed_time_ms = start.time_till(end)  # Час у мілісекундах

# Copy the result back to the CPU
c_cpu = c_gpu.get()
print(c_cpu)
print("Elapsed time (ms):", elapsed_time_ms)

# Clean up context
context = pycuda.autoinit.context

# Clean up
context.pop()