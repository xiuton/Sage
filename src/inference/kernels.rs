use burn::{prelude::*, tensor::backend::Backend};

/// FP8 activation quantization
pub fn act_quant<B: Backend>(input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
    // Calculate scale factor
    let max_val = input.clone().abs().max();
    let scale_tensor = Tensor::<B, 1, Float>::from_data([127.0], &input.device());
    let scale = max_val / scale_tensor;
    
    // Quantize to FP8
    let quantized = (input / scale.clone().unsqueeze()).round().clamp(-128.0, 127.0);
    
    (quantized, scale)
}

/// FP8 weight dequantization
pub fn weight_dequant<B: Backend>(weight: Tensor<B, 2>, scale_inv: Tensor<B, 1>) -> Tensor<B, 2> {
    weight * scale_inv.unsqueeze()
}

/// FP8 matrix multiplication
pub fn fp8_gemm<B: Backend>(a: Tensor<B, 2>, b: Tensor<B, 2>, scale_a: Tensor<B, 1>, scale_b: Tensor<B, 1>) -> Tensor<B, 2> {
    // Dequantize weights
    let b_dequant = b * scale_b.unsqueeze();
    
    // Perform matrix multiplication
    let result = a.matmul(b_dequant);
    
    // Apply output scale
    result * scale_a.unsqueeze()
}
