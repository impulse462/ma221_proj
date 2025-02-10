import numpy as np
import cupy as cp
import numpy.fft as fft
from numpy.fft import fft2, ifft2
import sigpy
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/batman/Documents/ma221/gfloat/src')
import pywt
import gfloat
print(dir(gfloat))
from gfloat.formats import *
from gfloat import decode_float
from scipy.linalg import cholesky

#-----------------low precision helper functions------------------
def to_fp8(x, format=format_info_binary32):
    fp8_real = gfloat.round_ndarray(format, x.real) #format_info_ocp_e4m3 
    fp8_imag = gfloat.round_ndarray(format, x.imag)
    fp8_complex = fp8_real + 1j * fp8_imag
    return fp8_complex

def to_fp16(x):
    real_part = x.real.astype(np.float16)
    imag_part = x.imag.astype(np.float16)
    return real_part + 1j * imag_part

def view_as_real(x):
    out_shape = x.shape + (2,)
    result = np.empty(out_shape, dtype=x.real.dtype)
    result[..., 0] = x.real
    result[..., 1] = x.imag
    return result


#----------------- Pad kspace and sensitivity maps for FP16 FFT on Tensor cores-------------
def pad(kspace, smaps): 
    num_coils, old_h, old_w = kspace.shape
    new_h, new_w = 256, 256

    # Initialize the new array with zeros
    kspace_padded = np.zeros((num_coils, new_h, new_w), dtype=kspace.dtype)
    start_h = (new_h - old_h) // 2  # e.g. (256 - 230)//2 = 13
    start_w = (new_w - old_w) // 2  # e.g. (256 - 180)//2 = 38

    # Place old data in the center
    kspace_padded[:, start_h:start_h+old_h, start_w:start_w+old_w] = kspace
    print(kspace_padded.shape)


    def zero_fill_in_k_space(sens_map):
        """
        sens_map: single-coil sensitivity map in the image domain (h, w).
        Returns: upsampled (256, 256) map.
        """
        h, w = kspace.shape[-2], kspace.shape[-1]
        new_h, new_w = 256, 256 # resize to 256 because that gets good speedup on FP16 FFT

        # 1) FFT
        sens_map_k = fft.fftshift(fft2(fft.ifftshift(sens_map)))

        # 2) Zero-fill in k-space
        sens_map_k_padded = np.zeros((new_h, new_w), dtype=sens_map_k.dtype)
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        sens_map_k_padded[start_h:start_h+h, start_w:start_w+w] = sens_map_k

        # 3) IFFT
        sens_map_padded = fft.fftshift(ifft2(fft.ifftshift(sens_map_k_padded)))

        return sens_map_padded

    # Apply this for each coil
    sens_maps_padded = np.zeros((num_coils, 256, 256), dtype=complex)
    for c in range(num_coils):
        sens_maps_padded[c] = zero_fill_in_k_space(smaps[c])

    return kspace_padded, sens_maps_padded


#------------------Proximal Gradient Descent in FP32 (default)----------------
def l1_wavelet_recon(
    kspace, 
    sensitivity_maps, 
    num_iters=100, 
    lamda=1e-2, 
    tol=1e-6, 
    use_fp16=False, 
    use_fp8=False, 
    format=format_info_binary32
    ):

    ishape = (kspace.shape[1], kspace.shape[2]) # image shape
    
    def get_weights(y):
        return (np.sqrt(np.sum(np.square(y), axis=0)) > 0).astype(y.dtype)
    
    W = get_weights(kspace)

    if use_fp16:
        W = to_fp16(W)
        kspace = to_fp16(kspace)
        sensitivity_maps = to_fp16(sensitivity_maps)
    elif use_fp8:
        W = to_fp8(W, format=format)
        kspace = to_fp8(kspace, format=format)
        sensitivity_maps = to_fp8(sensitivity_maps, format=format)

    def forward_op(x):
        #return A(x)
        """Forward operation: SENSE forward model."""
        shifted_image = np.fft.fftshift(x*sensitivity_maps, axes=(-2,-1)) # need to shift
        res =  np.fft.fft2(shifted_image, ishape, norm='ortho', axes=(-2,-1))
        res = W**0.5 * res

        if use_fp16:
            res = to_fp16(res)
        elif use_fp8:
            res = to_fp8(res, format=format)
        
        return res
    
    # x* FHSH
    def adjoint_op(y):
        #return A.H(y)
        """Adjoint operation: SENSE adjoint model."""
        ifft_result = np.fft.ifft2(W**0.5*y, ishape, norm='ortho',  axes=(-2,-1))
        shifted_ifft_result = np.fft.ifftshift(ifft_result, axes=(-2,-1)) # need to shift
        res = np.sum(np.conj(sensitivity_maps) * shifted_ifft_result, axis=0)

        if use_fp16:
            res = to_fp16(res)
        elif use_fp8:
            res = to_fp8(res, format=format)

        return res

    def soft_threshold(x, lamda):
        sign = np.where(np.abs(x) == 0, 0, x/np.abs(x))
        mag = np.abs(x) - lamda
        mag = (np.abs(mag) + mag)/2
        return sign * mag
    
    def g(x, alpha):
        
        # Wavelet transform
        zshape = [((i + 1) // 2) * 2 for i in x.shape]
        zinput = np.resize(x, zshape)
        coeffs = pywt.wavedecn(zinput, 'db4', mode='zero', axes=(-2,-1))
        w, coeff_slices = pywt.coeffs_to_array(coeffs, axes=(-2,-1))
        # proximal operator for L1 norm
        prox = soft_threshold(w, alpha*lamda)
        
        # Inverse Wavelet transform
        inv_input = pywt.array_to_coeffs(prox, coeff_slices, output_format='wavedecn')
        recn = pywt.waverecn(inv_input, 'db4', mode='zero', axes=(-2,-1))

        if use_fp16:
            recn = to_fp16(recn)
        elif use_fp8:
            recn = to_fp8(recn, format=format)

        return recn

    # Define AHA operator: A^H A + lambda I
    def normal_op(x):
        #return x + lamda * x
        """Normal equation operator: A^H A + lambda L1 norm"""
        result = adjoint_op(forward_op(x)) + lamda*x

        if use_fp16:
            return to_fp16(result)
        elif use_fp8:
            return to_fp8(result, format=format)
        
        return result
    
    b = adjoint_op(W**0.5 * kspace)
    x = np.zeros_like(b)  # 0s init
    r = np.zeros_like(b)  # 0s init

    alpha = 1
    residuals = []
    # gradient descent iterations
    for i in range(num_iters):
        x_old = x.copy()
        r = normal_op(x) - b + lamda*x 
        if use_fp16:
            r = to_fp16(normal_op(x) - b + lamda*x)
        elif use_fp8:
            r = to_fp8(normal_op(x) - b + lamda*x, format=format)
        else:   
            r = normal_op(x) - b + lamda*x
        
        x = g(x - alpha * r, alpha)
        resid = np.linalg.norm(x-x_old)/alpha
        
        if use_fp16:
            resid = to_fp16(resid)
        elif use_fp8:
            resid = to_fp8(resid, format=format)
        residuals.append(resid)


        if resid < tol:
            print(f'Converged after {i+1} iterations with residual {resid}')
            break

    return x


#---------------------- Proximal Gradient Descent in FP16 (cuFFT, TODO: cuBLAS for MV on tensor core)------------------
def l1_wavelet_recon_fp16(
    kspace, 
    sensitivity_maps, 
    num_iters=50, 
    lamda=1e-2, 
    tol=1e-6, 
    use_fp16=False, 
    use_fp8=False, 
    format=format_info_binary32
    ):

    ishape = (kspace.shape[1], kspace.shape[2]) # image shape
    num_coils, nrows, ncols = kspace.shape

    # Create FFT plans for tensor core operations
    shape = (num_coils, nrows, ncols)
    fft_plan = cp.cuda.cufft.XtPlanNd(
        shape[1:], shape[1:], 1, nrows*ncols, 'E', 
        shape[1:], 1, nrows*ncols, 'E',
        num_coils, 'E', order='C', last_axis=-1, last_size=None
    )
    
    ifft_plan = cp.cuda.cufft.XtPlanNd(
        shape[1:], shape[1:], 1, nrows*ncols, 'E',
        shape[1:], 1, nrows*ncols, 'E',
        num_coils, 'E', order='C', last_axis=-1, last_size=None
    )
    

    def get_weights(y):
        return (np.sqrt(np.sum(np.square(y), axis=0)) > 0).astype(y.dtype)
    
    W = get_weights(kspace)

    if use_fp16:
        W = to_fp16(W)
        kspace = to_fp16(kspace)
        sensitivity_maps = to_fp16(sensitivity_maps)
    elif use_fp8:
        W = to_fp8(W, format=format)
        kspace = to_fp8(kspace, format=format)
        sensitivity_maps = to_fp8(sensitivity_maps, format=format)

    def forward_op(x):
        #return A(x)

        # shifted_image = np.fft.fftshift(x*sensitivity_maps, axes=(-2,-1)) # need to shift
        #res =  np.fft.fft2(shifted_image, ishape, norm='ortho', axes=(-2,-1))
       
        shifted_image = np.fft.fftshift(x * sensitivity_maps, axes=(-2, -1))
        shifted_image = view_as_real(shifted_image)

        gpu_data = cp.array(shifted_image).astype(cp.float16)
    
        a = gpu_data.reshape(num_coils, nrows, 2*ncols)
        out = cp.empty_like(a)
        fft_plan.fft(a, out, cp.cuda.cufft.CUFFT_FORWARD)

        out = out.astype(cp.float32)
        out = out.reshape(num_coils, nrows, ncols, 2)
        out_complex = out.view(cp.complex64)
        out_complex = out_complex / np.sqrt(nrows * ncols) # normalize by 1/sqrt(N) for cufft_inverse
        res = cp.asnumpy(out_complex).squeeze(-1)
        res = W**0.5 * res

        if use_fp16:
            res = to_fp16(res)
        elif use_fp8:
            res = to_fp8(res, format=format)
        
        return res
    
    # x* FHSH
    def adjoint_op(y):
        #return A.H(y)

        # ifft_result = np.fft.ifftn(W**0.5*y, ishape, norm='ortho')
        # shifted_ifft_result = np.fft.ifftshift(ifft_result, axes=(-2,-1)) # need to shift
        # res = np.sum(np.conj(sensitivity_maps) * shifted_ifft_result, axis=0)
        
        gpu_pre = view_as_real(W**0.5 *y)
        gpu_data = cp.array(gpu_pre)
        gpu_real = gpu_data.astype(cp.float16)
        #print(num_coil, nrows, ncols)
        b = gpu_real.reshape(num_coils, nrows, 2*ncols)
        outb = cp.empty_like(b)
        
        # IFFT on tensor cores
    
        ifft_plan.fft(b, outb, cp.cuda.cufft.CUFFT_INVERSE)

        # Back to complex and CPU
        outb = outb.astype(cp.float32)
        outb = outb.reshape(num_coils, nrows, ncols, 2)
        outb_complex = outb.view(cp.complex64)
        
        outb_complex = outb_complex / np.sqrt(nrows * ncols) # normalize by 1/N for cufft_forward
        #print('adjoint:', np.min(outb_complex), np.max(outb_complex))

        outb_shift = cp.fft.ifftshift(outb_complex, axes=(1,2))
        ifft_shifted_result = cp.asnumpy(outb_shift).squeeze(-1)

        res = np.sum(np.conj(sensitivity_maps) * ifft_shifted_result, axis=0)

        if use_fp16:
            res = to_fp16(res)
        elif use_fp8:
            res = to_fp8(res, format=format)

        return res

    def soft_threshold(x, lamda):
        sign = np.where(np.abs(x) == 0, 0, x/np.abs(x))
        mag = np.abs(x) - lamda
        mag = (np.abs(mag) + mag)/2
        return sign * mag
    
    def g(x, alpha):
        
        # Wavelet transform
        zshape = [((i + 1) // 2) * 2 for i in x.shape]
        zinput = np.resize(x, zshape)
        coeffs = pywt.wavedecn(zinput, 'db4', mode='zero', axes=(-2,-1))
        w, coeff_slices = pywt.coeffs_to_array(coeffs, axes=(-2,-1))
        # proximal operator for L1 norm
        prox = soft_threshold(w, alpha*lamda)
        
        # Inverse Wavelet transform
        inv_input = pywt.array_to_coeffs(prox, coeff_slices, output_format='wavedecn')
        recn = pywt.waverecn(inv_input, 'db4', mode='zero', axes=(-2,-1))

        if use_fp16:
            recn = to_fp16(recn)
        elif use_fp8:
            recn = to_fp8(recn, format=format)

        return recn

    # Define AHA operator: A^H A + lambda I
    def normal_op(x):
        #return x + lamda * x
        """Normal equation operator: A^H A + lambda L1 norm"""
        result = adjoint_op(forward_op(x)) + lamda*x

        if use_fp16:
            return to_fp16(result)
        elif use_fp8:
            return to_fp8(result, format=format)
        
        return result
    
    b = adjoint_op(W**0.5 * kspace)
    x = np.zeros_like(b)  # 0s init
    r = np.zeros_like(b)  # 0s init

    alpha = 1
    residuals = []
    # gradient descent iterations
    for i in range(num_iters):
        x_old = x.copy()
        r = normal_op(x) - b + lamda*x 
        if use_fp16:
            r = to_fp16(normal_op(x) - b + lamda*x)
        elif use_fp8:
            r = to_fp8(normal_op(x) - b + lamda*x, format=format)
        else:   
            r = normal_op(x) - b + lamda*x
        
        x = g(x - alpha * r, alpha)
        resid = np.linalg.norm(x-x_old)/alpha
        
        if use_fp16:
            resid = to_fp16(resid)
        elif use_fp8:
            resid = to_fp8(resid, format=format)
        residuals.append(resid)


        if resid < tol:
            print(f'Converged after {i+1} iterations with residual {resid}')
            break

    return x