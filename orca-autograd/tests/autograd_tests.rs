use orca_autograd::{Autodiff, AutogradTensorExt};
use orca_backend_cpu::CpuBackend;
use orca_core::{Result, Shape};
use orca_tensor::Tensor;

#[test]
fn test_basic_autograd() -> Result<()> {
    let backend = CpuBackend;
    let autodiff_backend = Autodiff::new(backend);

    // Create leaf tensors
    let mut x = Tensor::from_f32_slice(
        autodiff_backend.clone(),
        &[2.0, 3.0],
        Shape::new(vec![1, 2]),
    )?;
    let mut w = Tensor::from_f32_slice(
        autodiff_backend.clone(),
        &[1.0, 1.0],
        Shape::new(vec![2, 1]),
    )?;

    x.require_grad();
    w.require_grad();

    // y = x @ w
    let y = x.matmul(&w)?;
    assert_eq!(y.shape().to_vec(), vec![1, 1]);

    // loss = (y - 10)^2
    let ten = Tensor::from_f32_slice(autodiff_backend.clone(), &[10.0], Shape::new(vec![1, 1]))?;
    let diff = (&y - &ten)?;
    let loss = (&diff * &diff)?;

    // Run backward
    loss.backward()?;

    // Check gradients
    let grad_w = w.grad().expect("Should have gradient");
    let grad_w_data = grad_w.primal().to_bytes()?;
    let grad_w_floats: &[f32] = bytemuck::cast_slice(&grad_w_data);

    // y = 2*1 + 3*1 = 5.
    // diff = 5 - 10 = -5.
    // loss = diff^2 = 25.
    // dloss/dy = 2 * diff = -10.
    // dy/dw = x^T = [[2], [3]].
    // dloss/dw = -10 * [[2], [3]] = [[-20], [-30]].
    assert!((grad_w_floats[0] - (-20.0)).abs() < 1e-5);
    assert!((grad_w_floats[1] - (-30.0)).abs() < 1e-5);

    Ok(())
}

#[test]
fn test_autograd_activation() -> Result<()> {
    let backend = CpuBackend;
    let autodiff_backend = Autodiff::new(backend);

    let mut x = Tensor::from_f32_slice(
        autodiff_backend.clone(),
        &[-1.0, 2.0],
        Shape::new(vec![1, 2]),
    )?;
    x.require_grad();

    let y = x.relu()?;
    let y_data = y.primal().to_bytes()?;
    let y_floats: &[f32] = bytemuck::cast_slice(&y_data);
    assert_eq!(y_floats[0], 0.0);
    assert_eq!(y_floats[1], 2.0);

    let ones = Tensor::from_f32_slice(
        autodiff_backend.clone(),
        &[1.0, 1.0],
        Shape::new(vec![1, 2]),
    )?;
    let loss = (&y * &ones)?;
    loss.backward()?;

    let grad_x = x.grad().expect("Should have gradient");
    let grad_x_data = grad_x.primal().to_bytes()?;
    let grad_x_floats: &[f32] = bytemuck::cast_slice(&grad_x_data);

    // grad for negative input should be 0, positive input should be 1
    assert_eq!(grad_x_floats[0], 0.0);
    assert_eq!(grad_x_floats[1], 1.0);

    Ok(())
}
