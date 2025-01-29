import cupy as cp
import numpy as np
import time
from typing import List, Tuple

def create_complex_input(height: int, width: int, dtype=cp.float16) -> cp.ndarray:
    """Create a complex-valued input array stored as pairs of float16"""
    return cp.random.random((height, 2*width)).astype(dtype)

def benchmark_fft2d_half(height: int, width: int, num_iterations: int = 100) -> Tuple[List[float], cp.ndarray, cp.ndarray]:
    """Benchmark 2D FFT using half precision"""
    shape = (height, width)
    
    print("\nDebug info:")
    print(f"Shape: {shape}")
    print(f"Total elements: {height * width}")
    
    # Create input and output arrays
    a = create_complex_input(height, width, cp.float16)
    out = cp.empty_like(a)
    
    print(f"Input array shape: {a.shape}")
    print(f"Input array dtype: {a.dtype}")
    
    try:
        # Create plan for 2D FFT with updated parameters for CuPy 10.4.0
        # Create plan for 2D FFT following the correct parameter structure
        plan = cp.cuda.cufft.XtPlanNd(shape,           # transform shape
                                     shape, 1, height*width, 'E',  # input params
                                     shape, 1, height*width, 'E',  # output params
                                     1, 'E',                      # batch size and type
                                     order='C', last_axis=-1, last_size=None)
        
        print("FFT plan created successfully")
        
        # Warmup
        for _ in range(5):
            plan.fft(a, out, cp.cuda.cufft.CUFFT_FORWARD)
        cp.cuda.stream.get_current_stream().synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            cp.cuda.stream.get_current_stream().synchronize()
            start = time.perf_counter()
            plan.fft(a, out, cp.cuda.cufft.CUFFT_FORWARD)
            cp.cuda.stream.get_current_stream().synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
            
        return times, a, out
        
    except Exception as e:
        print(f"Error creating or executing FFT plan: {str(e)}")
        raise

def benchmark_fft2d_single(height: int, width: int, num_iterations: int = 100) -> Tuple[List[float], cp.ndarray, cp.ndarray]:
    """Benchmark regular single precision 2D FFT"""
    a = (cp.random.random((height, width)) + 
         1j * cp.random.random((height, width))).astype(cp.complex64)
    
    # Warmup
    for _ in range(5):
        _ = cp.fft.fft2(a)
    cp.cuda.stream.get_current_stream().synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        cp.cuda.stream.get_current_stream().synchronize()
        start = time.perf_counter()
        _ = cp.fft.fft2(a)
        cp.cuda.stream.get_current_stream().synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return times, a, cp.fft.fft2(a)

def print_statistics(times: List[float], label: str) -> None:
    """Print detailed timing statistics"""
    print(f"\n{label} Statistics:")
    print(f"Mean time: {np.mean(times):.3f} ms")
    print(f"Median time: {np.median(times):.3f} ms")
    print(f"Std dev: {np.std(times):.3f} ms")
    print(f"Min time: {np.min(times):.3f} ms")
    print(f"Max time: {np.max(times):.3f} ms")
    print(f"95th percentile: {np.percentile(times, 95):.3f} ms")

def run_comparison(sizes: List[Tuple[int, int]], num_iterations: int = 100) -> None:
    """Run comprehensive comparison between FP16 and FP32 FFTs"""
    print(f"CuPy version: {cp.__version__}")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}\n")
    
    for height, width in sizes:
        print(f"\nImage size: {height}x{width}")
        print("=" * 50)
        
        # Memory usage calculation
        fp32_memory = height * width * 8  # complex64 = 8 bytes
        fp16_memory = height * width * 4  # complex32 = 4 bytes
        print(f"Memory usage per array:")
        print(f"FP32: {fp32_memory/1024/1024:.2f} MB")
        print(f"FP16: {fp16_memory/1024/1024:.2f} MB")
        
        # Run benchmarks
        print("\nRunning benchmarks...")
        
        # Try FP16 benchmark first
        try:
            half_times, half_in, half_out = benchmark_fft2d_half(height, width, num_iterations)
            print_statistics(half_times, "FP16 FFT")
        except Exception as e:
            print(f"\nFP16 benchmark failed: {str(e)}")
            half_times = None
        
        # Run FP32 benchmark
        single_times, single_in, single_out = benchmark_fft2d_single(height, width, num_iterations)
        print_statistics(single_times, "FP32 FFT")
        
        # Calculate speedup if both benchmarks succeeded
        if half_times is not None:
            speedup = np.mean(single_times) / np.mean(half_times)
            print(f"\nSpeedup (FP32/FP16): {speedup:.2f}x")
        
        # Clear memory
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        
        print("\n" + "="*50)

if __name__ == "__main__":
    sizes = [
        (64, 64), 
        (128, 128), 
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192)
    ]
    
    num_iterations = 100
    
    try:
        run_comparison(sizes, num_iterations)
    except Exception as e:
        print(f"\nError during benchmark: {str(e)}")
        print("\nDebug information:")
        print(f"CuPy version: {cp.__version__}")
        print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        print("Available GPU memory:", cp.cuda.runtime.memGetInfo()[0] / 1024**2, "MB")