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
#print(dir(gfloat))
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




#------------------Conjugate Gradient in FP32 (default)----------------
def cg_recon(
    kspace, 
    sensitivity_maps, 
    num_iters=10, 
    lamda=1e-2, 
    tol=1e-6, 
    use_fp16=False, 
    use_fp8=False,
    use_scaling=False, 
    format=format_info_binary32
    ):
    num_coil, nrows, ncols = kspace.shape

    def get_weights(y):
        # here y is already undersampled via poisson disc
        return (np.sqrt(np.sum(np.square(y), axis=0)) > 0).astype(y.dtype)

    W = get_weights(kspace)

    def forward_op(x): # W^1/2*F*S*x
        x_expand = np.expand_dims(x, axis=0)
        shifted_image = np.fft.fftshift(sensitivity_maps * x_expand, axes=(1, 2))
        inter = np.fft.fft2(shifted_image)
        return W**0.5 * inter #np.fft.fft2(shifted_image)#, norm='ortho')

    def adjoint_op(y):
        ifft_result = np.fft.ifft2(y, norm='ortho') #W**0.5 * y?
        shifted_ifft_result = np.fft.ifftshift(ifft_result, axes=(1, 2))
        return np.sum(sensitivity_maps.conj() * shifted_ifft_result, axis=0)

    def normal_op(x):
        result = adjoint_op(forward_op(x)) + lamda * x
        if use_fp8:
            return to_fp8(result, format)  # Consider applying FP8 here sparingly
        elif use_fp16:
            return to_fp16(result, format)
        else:
            return result

    def estimate_condition_number(normal_op, x_shape, num_power_iters=20):
        v1 = np.random.randn(*x_shape) + 1j * np.random.randn(*x_shape)
        v2 = np.random.randn(*x_shape) + 1j * np.random.randn(*x_shape)
        
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # power iteration for largest eigenvalue
        for _ in range(num_power_iters):
            v1_new = normal_op(v1)
            v1 = v1_new / np.linalg.norm(v1_new)
        
        lambda_max = np.real(np.vdot(v1, normal_op(v1)))
        lambda_min = lamda  # here lambda is just lower bound for condition number
        
        return lambda_max / lambda_min, lambda_max, lambda_min

    # calculate condition number before starting CG
    cond_num, max_eval, min_eval = estimate_condition_number(normal_op, (nrows, ncols))
    print(f"Estimated condition number: {cond_num}")
    print(f"Largest eigenvalue: {max_eval}")
    print(f"Smallest eigenvalue: {min_eval}")


    if use_fp8:
        AHy = to_fp8(adjoint_op(kspace), format)
    elif use_fp16:
        AHy = to_fp16(adjoint_op(kspace))
    else:
        AHy = adjoint_op(kspace)
    

    # Initialize variables in FP16 or FP32
    x = np.zeros_like(AHy, dtype=np.float16 if use_fp16 else np.complex64)
    r = AHy - normal_op(x)


    # scaling
    scale_factor = 1.0
    if use_scaling:
        scale_factor = max(np.linalg.norm(r), 1e-6)
        r /= scale_factor
 
    z = r.copy()
    p = z.copy()

    rz_old = np.vdot(r, z)
    resid = np.sqrt(rz_old)
    residuals = [resid]

    for i in range(num_iters):
        # Only cast Ap to FP8 if necessary
        if use_fp8:
            Ap = to_fp8(normal_op(p), format)
        elif use_fp16:
            Ap = to_fp16(normal_op(p))
        else:
            Ap = normal_op(p)

        pAp = np.vdot(p, Ap) #TODO p_fp8 here
        #=========== check ================
        # print(f"Iteration {i}:")
        # print(f"pAp = {pAp}")
        # print(f"rz_old = {rz_old}")

        if pAp <= 0:
            print("Matrix not positive definite. Stopping.")
            break

        alpha = rz_old / pAp
        if use_scaling:
            x += alpha * p * scale_factor
        else:
            x += alpha * p
        r -= alpha * Ap



        # recompute scaling factor here (dynamic scaling)
        if use_scaling:
            scale_factor = max(np.linalg.norm(r), 1e-6)
            r /= scale_factor

        if use_fp8:
            z = to_fp8(r, format) * scale_factor  # (reduce memory here? #TODO remultiply scale factor
        elif use_fp16:
            z = to_fp16(r)
        else:
            z = r

        rz_new = np.vdot(r, z) # try r_fp8
        resid = np.sqrt(rz_new)
        residuals.append(resid)

        if resid < tol:
            print(f'Converged after {i+1} iterations with residual {resid}')
            break

        beta = rz_new / rz_old
        if use_fp8:
            p = to_fp8(z + beta * p, format)
        elif use_fp16:
            p = to_fp16(z + beta * p)
        else:
            p = z + beta * p

        rz_old = rz_new
    return x


#---------------------- Conjugate Gradient in FP16 (cuFFT, TODO: cuBLAS for MV on tensor core)------------------
"""
Inputs: complex-valued kspace dimension (num_coil, h, w),
        complex-valued smaps dimension (num_coil, h, w)
Outpus: complex-valued reconstructed image (h,w)

"""
def cg_sense_recon_fp16(
    kspace, 
    sensitivity_maps, 
    num_iters=1, 
    lamda=1e-2, 
    tol=1e-6, 
    use_fp16=False, 
    use_fp8=False,
    use_scaling=False, 
    format=format_info_binary32
    ):

    num_coils, nrows, ncols = kspace.shape
        
    # make sure dimensions are powers of 2    
    assert (nrows & (nrows-1)) == 0
    assert (ncols & (nrows-1)) == 0

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
        return (np.sqrt(np.sum(np.square(np.abs(y)), axis=0)) > 0).astype(y.dtype)

    # Keep weights in original precision for now (complex64, shape: (256,256))
    W = get_weights(kspace)

    def forward_op(x):  # W^1/2 * F * S * x

        # x_expand = np.expand_dims(x, axis=0)
        # shifted_image = np.fft.fftshift(sensitivity_maps * x_expand, axes=(1, 2))
        # inter = np.fft.fft2(shifted_image)
        # print('forward:', np.min(inter), np.max(inter))
        # return W**0.5 * inter
        

        x_expand = np.expand_dims(x, axis=0)
       
        shifted_image = np.fft.fftshift(sensitivity_maps * x_expand, axes=(1, 2))

        # prepare for FFT on tensor core
        shifted_image = view_as_real(shifted_image)
        gpu_data = cp.array(shifted_image).astype(cp.float16) # cast to fp16 for tensor core
        a = gpu_data.reshape(num_coils, nrows, 2*ncols)
        out = cp.empty_like(a)
        fft_plan.fft(a, out, cp.cuda.cufft.CUFFT_FORWARD) # execute fft plan
        out = out.astype(cp.float32) # recast to fp32 for accumulate
        out = out.reshape(num_coils, nrows, ncols, 2)
        out_complex = out.view(cp.complex64) # no normalization needed for cufft_forward
        result = cp.asnumpy(out_complex).squeeze(-1)
        #--------end tensor core FFT-----------

        return W**0.5 * result

    def adjoint_op(y): # y * W^1/2 * F^H * S^H
        # ifft_result = np.fft.ifft2(y, norm='ortho') #W**0.5 * y?
        # print('adjoint:', np.min(ifft_result), np.max(ifft_result))
        # shifted_ifft_result = np.fft.ifftshift(ifft_result, axes=(1, 2))
        
        # Prep for IFFT on tensor core
        gpu_pre = view_as_real(y * W**0.5)
        gpu_data = cp.array(gpu_pre)
        gpu_real = gpu_data.astype(cp.float16) # here cast to float16 for tensor computation
        b = gpu_real.reshape(num_coils, nrows, 2*ncols)
        outb = cp.empty_like(b)
        ifft_plan.fft(b, outb, cp.cuda.cufft.CUFFT_INVERSE) # execute ifft plan
        #------------end tensor core FFT----------

        # back to complex and CPU
        outb = outb.astype(cp.float32)
        outb = outb.reshape(num_coils, nrows, ncols, 2)
        outb_complex = outb.view(cp.complex64)
        
        outb_complex = outb_complex / np.sqrt(nrows * ncols) # !!!normalize by 1/sqrt(N) for cufft_inverse!!!
        print('adjoint:', np.min(outb_complex), np.max(outb_complex))

        outb_shift = cp.fft.ifftshift(outb_complex, axes=(1,2)) # axes need to be specified here, otherwise bad image
        shifted_ifft_result = cp.asnumpy(outb_shift).squeeze(-1)

        return np.sum(sensitivity_maps.conj() * shifted_ifft_result, axis=0)


    def normal_op(x):
        result = adjoint_op(forward_op(x)) + lamda * x
        if use_fp8:
            return to_fp8(result, format)
        elif use_fp16:
            return to_fp16(result)
        else:
            return result

    if use_fp8:
        AHy = to_fp8(adjoint_op(kspace), format)
    elif use_fp16:
        AHy = to_fp16(adjoint_op(kspace))
    else:
        AHy = adjoint_op(kspace)

    # initialize variables in FP16 or FP32
    x = np.zeros_like(AHy, dtype=np.float16 if use_fp16 else np.complex64)
    r = AHy - normal_op(x)


    # scaling
    scale_factor = 1.0
    if use_scaling:
        scale_factor = max(np.linalg.norm(r), 1e-6)
        r /= scale_factor

    z = r.copy()
    p = z.copy()

    rz_old = np.vdot(r, z)
    resid = np.sqrt(rz_old)
    residuals = [resid]

    for i in range(num_iters):
        # Only cast Ap to FP8 if necessary
        if use_fp8:
            Ap = to_fp8(normal_op(p), format)
        elif use_fp16:
            Ap = to_fp16(normal_op(p))
        else:
            Ap = normal_op(p)

        pAp = np.vdot(p, Ap) #TODO p_fp8 here

        #=========== check ================
        # print(f"Iteration {i}:")
        # print(f"pAp = {pAp}")
        # print(f"rz_old = {rz_old}")
    
        if pAp <= 0:
            print("Matrix not positive definite. Stopping.")
            break

        alpha = rz_old / pAp
        #alpha = alpha * min(1.0, abs(pAp)/abs(rz_old))
        if use_scaling:
            x += alpha * p * scale_factor
        else:
            x += alpha * p    
        r -= alpha * Ap


        # recompute scaling factor here (dynamic scaling)
        if use_scaling:
            scale_factor = max(np.linalg.norm(r), 1e-6)
            r /= scale_factor

        if use_fp8:
            z = to_fp8(r, format) * scale_factor  # (reduce memory here? #TODO remultiply scale factor
        elif use_fp16:
            z = to_fp16(r)
        else:
            z = r

        rz_new = np.vdot(r, z) # try r_fp8
        resid = np.sqrt(rz_new)
        residuals.append(resid)

        if resid < tol:
            print(f'Converged after {i+1} iterations with residual {resid}')
            break

        if np.abs(rz_old) < 1e-15:
            print("rz_old too small, stopping iterations")
            break

        beta = rz_new / rz_old
        
        if use_fp8:
            p = to_fp8(z + beta * p, format)
        elif use_fp16:
            p = to_fp16(z + beta * p)
        else:
            p = z + beta * p

        rz_old = rz_new
        if i % 10 == 0:
            print("Iteration:", i) 
    return x
    
    
    cp.get_default_memory_pool().free_all_blocks()
    
    return x