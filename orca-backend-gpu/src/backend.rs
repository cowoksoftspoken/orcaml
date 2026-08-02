#![allow(unused_variables, dead_code)]

use std::sync::Arc;

use crate::storage::GpuStorage;
use orca_backend_cpu::CpuBackend;
use orca_core::{DType, Device, OrcaError, Result, Shape};
use orca_tensor::Backend;
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct Pipelines {
    pub add: wgpu::ComputePipeline,
    pub sub: wgpu::ComputePipeline,
    pub mul: wgpu::ComputePipeline,
    pub div: wgpu::ComputePipeline,
    pub relu: wgpu::ComputePipeline,
    pub sqrt: wgpu::ComputePipeline,
    pub sigmoid: wgpu::ComputePipeline,
    pub exp: wgpu::ComputePipeline,
    pub log: wgpu::ComputePipeline,
    pub relu_bw: wgpu::ComputePipeline,
    pub sigmoid_bw: wgpu::ComputePipeline,
    pub exp_bw: wgpu::ComputePipeline,
    pub log_bw: wgpu::ComputePipeline,
    pub mul_scalar: wgpu::ComputePipeline,
    pub matmul: wgpu::ComputePipeline,
    pub expand: wgpu::ComputePipeline,
    pub sum_to_shape: wgpu::ComputePipeline,
    pub transpose: wgpu::ComputePipeline,
    pub conv2d: wgpu::ComputePipeline,
    pub conv2d_bw_input: wgpu::ComputePipeline,
    pub conv2d_bw_weight: wgpu::ComputePipeline,
    pub conv2d_bw_bias: wgpu::ComputePipeline,
    pub check_nan_inf: wgpu::ComputePipeline,
    pub gather: wgpu::ComputePipeline,
    pub scatter: wgpu::ComputePipeline,
    pub scatter_bw_base: wgpu::ComputePipeline,
    pub gather_bw: wgpu::ComputePipeline,
    pub max_to_shape: wgpu::ComputePipeline,
    pub max_to_shape_bw: wgpu::ComputePipeline,
}

#[derive(Clone, Debug)]
pub struct GpuBackend {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub pipelines: Arc<Pipelines>,
    pub pool: Arc<std::sync::Mutex<crate::pool::MemoryPool>>,
}

impl GpuBackend {
    pub fn new() -> Result<Self> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .ok_or_else(|| {
                    OrcaError::InternalError("Failed to find a suitable GPU adapter".into())
                })?;

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        required_features: wgpu::Features::empty(),
                        required_limits: adapter.limits(),
                        label: None,
                    },
                    None,
                )
                .await
                .map_err(|err| {
                    OrcaError::InternalError(format!("Failed to create GPU device: {err}"))
                })?;

            // Compile Shaders
            let binary_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("binary"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/binary.wgsl").into()),
            });
            let unary_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("unary"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/unary.wgsl").into()),
            });
            let grad_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("grad"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/grad.wgsl").into()),
            });
            let scalar_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scalar"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scalar.wgsl").into()),
            });
            let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
            });
            let broadcast_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("broadcast"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/broadcast.wgsl").into()),
            });
            let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("reduce"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/reduce.wgsl").into()),
            });
            let transpose_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("transpose"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/transpose.wgsl").into()),
            });
            let conv2d_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("conv2d"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/conv2d.wgsl").into()),
            });
            let conv2d_bw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("conv2d_bw"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/conv2d_bw.wgsl").into()),
            });
            let check_nan_inf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("check_nan_inf"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/check_nan_inf.wgsl").into()),
            });
            let gather_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gather"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gather.wgsl").into()),
            });
            let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scatter"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scatter.wgsl").into()),
            });
            let scatter_bw_base_shader =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("scatter_bw_base"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("shaders/scatter_backward_base.wgsl").into(),
                    ),
                });
            let gather_bw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gather_bw"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/gather_backward.wgsl").into(),
                ),
            });
            let max_to_shape_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("max_to_shape"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/reduce_max.wgsl").into()),
            });
            let max_to_shape_bw_shader =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("max_to_shape_bw"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("shaders/reduce_max_backward.wgsl").into(),
                    ),
                });

            let create_pipeline = |shader: &wgpu::ShaderModule, entry: &str| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: shader,
                    entry_point: entry,
                    compilation_options: Default::default(),
                })
            };

            let pipelines = Pipelines {
                add: create_pipeline(&binary_shader, "add_main"),
                sub: create_pipeline(&binary_shader, "sub_main"),
                mul: create_pipeline(&binary_shader, "mul_main"),
                div: create_pipeline(&binary_shader, "div_main"),
                relu: create_pipeline(&unary_shader, "relu_main"),
                sqrt: create_pipeline(&unary_shader, "sqrt_main"),
                sigmoid: create_pipeline(&unary_shader, "sigmoid_main"),
                exp: create_pipeline(&unary_shader, "exp_main"),
                log: create_pipeline(&unary_shader, "log_main"),
                relu_bw: create_pipeline(&grad_shader, "relu_backward_main"),
                sigmoid_bw: create_pipeline(&grad_shader, "sigmoid_backward_main"),
                exp_bw: create_pipeline(&grad_shader, "exp_backward_main"),
                log_bw: create_pipeline(&grad_shader, "log_backward_main"),
                mul_scalar: create_pipeline(&scalar_shader, "mul_scalar_main"),
                matmul: create_pipeline(&matmul_shader, "matmul_main"),
                expand: create_pipeline(&broadcast_shader, "expand_main"),
                sum_to_shape: create_pipeline(&reduce_shader, "sum_to_shape_main"),
                transpose: create_pipeline(&transpose_shader, "transpose_main"),
                conv2d: create_pipeline(&conv2d_shader, "forward_main"),
                conv2d_bw_input: create_pipeline(&conv2d_bw_shader, "backward_input_main"),
                conv2d_bw_weight: create_pipeline(&conv2d_bw_shader, "backward_weight_main"),
                conv2d_bw_bias: create_pipeline(&conv2d_bw_shader, "backward_bias_main"),
                check_nan_inf: create_pipeline(&check_nan_inf_shader, "check_main"),
                gather: create_pipeline(&gather_shader, "gather_main"),
                scatter: create_pipeline(&scatter_shader, "scatter_main"),
                scatter_bw_base: create_pipeline(&scatter_bw_base_shader, "scatter_bw_base_main"),
                gather_bw: create_pipeline(&gather_bw_shader, "gather_bw_main"),
                max_to_shape: create_pipeline(&max_to_shape_shader, "max_to_shape_main"),
                max_to_shape_bw: create_pipeline(&max_to_shape_bw_shader, "max_to_shape_bw_main"),
            };

            let pool = Arc::new(std::sync::Mutex::new(crate::pool::MemoryPool::new()));
            Ok(Self {
                device: Arc::new(device),
                queue: Arc::new(queue),
                pipelines: Arc::new(pipelines),
                pool,
            })
        })
    }

    /// Allocates a buffer using the caching MemoryPool.
    pub fn allocate_buffer(
        &self,
        size: wgpu::BufferAddress,
        mut usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        if usage.contains(wgpu::BufferUsages::STORAGE) {
            usage |= wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        }
        let mut pool = match self.pool.lock() {
            Ok(pool) => pool,
            Err(poisoned) => poisoned.into_inner(),
        };
        pool.allocate(&self.device, size, usage)
    }

    fn execute_unary(
        &self,
        pipeline: &wgpu::ComputePipeline,
        storage: &GpuStorage,
        shape: &Shape,
    ) -> Result<GpuStorage> {
        let num_elements = storage.num_elements;
        let out_buffer = self.allocate_buffer(
            (num_elements * storage.element_size) as wgpu::BufferAddress,
            wgpu::BufferUsages::STORAGE,
        );

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((num_elements as f32) / 64.0).ceil() as u32;
            cpass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(GpuStorage::new(
            out_buffer,
            num_elements,
            storage.element_size,
            self.pool.clone(),
        ))
    }

    fn execute_binary(
        &self,
        pipeline: &wgpu::ComputePipeline,
        lhs: &GpuStorage,
        rhs: &GpuStorage,
        shape: &Shape,
    ) -> Result<GpuStorage> {
        if lhs.num_elements != rhs.num_elements {
            return Err(OrcaError::ShapeMismatch {
                op: "binary_gpu",
                expected: lhs.num_elements.to_string(),
                got: rhs.num_elements.to_string(),
            });
        }
        let num_elements = lhs.num_elements;
        let out_buffer = self.allocate_buffer(
            (num_elements * lhs.element_size) as wgpu::BufferAddress,
            wgpu::BufferUsages::STORAGE,
        );

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((num_elements as f32) / 64.0).ceil() as u32;
            cpass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(GpuStorage::new(
            out_buffer,
            num_elements,
            lhs.element_size,
            self.pool.clone(),
        ))
    }
}

impl Backend for GpuBackend {
    type Storage = GpuStorage;

    fn device(&self) -> Device {
        Device::Gpu(0)
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let element_size = 4; // Assuming f32 for now
        let num_elements = shape.num_elements();
        let size = (num_elements * element_size) as wgpu::BufferAddress;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zeros"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(GpuStorage::new(
            buffer,
            num_elements,
            element_size,
            self.pool.clone(),
        ))
    }

    fn from_f32_slice(&self, shape: &Shape, data: &[f32]) -> Result<Self::Storage> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("from_f32"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        Ok(GpuStorage::new(
            buffer,
            shape.num_elements(),
            4,
            self.pool.clone(),
        ))
    }

    fn to_f32_vec(&self, storage: &Self::Storage) -> Result<Vec<f32>> {
        let size = (storage.num_elements * storage.element_size) as wgpu::BufferAddress;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(storage.buffer()?, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });
        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = receiver.recv() {
            let data = slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging.unmap();
            Ok(result)
        } else {
            Err(OrcaError::InternalError("GPU read failed".to_string()))
        }
    }

    fn add(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        self.execute_binary(&self.pipelines.add, lhs, rhs, shape)
    }

    fn sub(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        self.execute_binary(&self.pipelines.sub, lhs, rhs, shape)
    }

    fn mul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        self.execute_binary(&self.pipelines.mul, lhs, rhs, shape)
    }

    fn div(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        self.execute_binary(&self.pipelines.div, lhs, rhs, shape)
    }

    fn relu(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.execute_unary(&self.pipelines.relu, storage, shape)
    }

    fn sqrt(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.execute_unary(&self.pipelines.sqrt, storage, shape)
    }

    fn sigmoid(
        &self,
        storage: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        self.execute_unary(&self.pipelines.sigmoid, storage, shape)
    }

    fn exp(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.execute_unary(&self.pipelines.exp, storage, shape)
    }

    fn log(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.execute_unary(&self.pipelines.log, storage, shape)
    }

    fn relu_backward(
        &self,
        grad_out: &Self::Storage,
        primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        self.execute_binary(&self.pipelines.relu_bw, grad_out, primal, shape)
    }

    fn sigmoid_backward(
        &self,
        grad_out: &Self::Storage,
        primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        self.execute_binary(&self.pipelines.sigmoid_bw, grad_out, primal, shape)
    }

    fn exp_backward(
        &self,
        grad_out: &Self::Storage,
        primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        self.execute_binary(&self.pipelines.exp_bw, grad_out, primal, shape)
    }

    fn log_backward(
        &self,
        grad_out: &Self::Storage,
        primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        self.execute_binary(&self.pipelines.log_bw, grad_out, primal, shape)
    }

    fn div_backward_lhs(
        &self,
        grad_out: &Self::Storage,
        rhs_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let cpu = CpuBackend;
        let grad = cpu.from_f32_slice(shape, &self.to_f32_vec(grad_out)?)?;
        let r = cpu.from_f32_slice(shape, &self.to_f32_vec(rhs_primal)?)?;
        let out = cpu.div_backward_lhs(&grad, &r, shape, dtype)?;
        self.from_f32_slice(shape, &cpu.to_f32_vec(&out)?)
    }

    fn div_backward_rhs(
        &self,
        grad_out: &Self::Storage,
        lhs_primal: &Self::Storage,
        rhs_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let cpu = CpuBackend;
        let grad = cpu.from_f32_slice(shape, &self.to_f32_vec(grad_out)?)?;
        let l = cpu.from_f32_slice(shape, &self.to_f32_vec(lhs_primal)?)?;
        let r = cpu.from_f32_slice(shape, &self.to_f32_vec(rhs_primal)?)?;
        let out = cpu.div_backward_rhs(&grad, &l, &r, shape, dtype)?;
        self.from_f32_slice(shape, &cpu.to_f32_vec(&out)?)
    }

    fn sqrt_backward(
        &self,
        grad_out: &Self::Storage,
        out_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let cpu = CpuBackend;
        let grad = cpu.from_f32_slice(shape, &self.to_f32_vec(grad_out)?)?;
        let out_p = cpu.from_f32_slice(shape, &self.to_f32_vec(out_primal)?)?;
        let out_res = cpu.sqrt_backward(&grad, &out_p, shape, dtype)?;
        self.from_f32_slice(shape, &cpu.to_f32_vec(&out_res)?)
    }

    fn accumulate_grad(&self, lhs: &Self::Storage, rhs: &Self::Storage) -> Result<Self::Storage> {
        let shape = Shape::new(vec![lhs.num_elements]);
        self.execute_binary(&self.pipelines.add, lhs, rhs, &shape)
    }

    fn conv2d(
        &self,
        input: &Self::Storage,
        weight: &Self::Storage,
        bias: Option<&Self::Storage>,
        in_shape: &Shape,
        weight_shape: &Shape,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let n = in_shape[0] as u32;
        let c_in = in_shape[1] as u32;
        let h_in = in_shape[2] as u32;
        let w_in = in_shape[3] as u32;
        let c_out = weight_shape[0] as u32;
        let k_h = weight_shape[2] as u32;
        let k_w = weight_shape[3] as u32;
        let h_out =
            (h_in + 2 * padding as u32 - (dilation as u32 * (k_h - 1) + 1)) / stride as u32 + 1;
        let w_out =
            (w_in + 2 * padding as u32 - (dilation as u32 * (k_w - 1) + 1)) / stride as u32 + 1;

        let num_elements = (n * c_out * h_out * w_out) as usize;
        let out_buffer = self.allocate_buffer(
            (num_elements * 4) as wgpu::BufferAddress,
            wgpu::BufferUsages::STORAGE,
        );

        let mut uniforms = [0u32; 16];
        uniforms[0] = n;
        uniforms[1] = c_in;
        uniforms[2] = h_in;
        uniforms[3] = w_in;
        uniforms[4] = c_out;
        uniforms[5] = k_h;
        uniforms[6] = k_w;
        uniforms[7] = h_out;
        uniforms[8] = w_out;
        uniforms[9] = padding as u32;
        uniforms[10] = stride as u32;
        uniforms[11] = if bias.is_some() { 1 } else { 0 };
        uniforms[12] = dilation as u32;
        uniforms[13] = groups as u32;

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("conv2d_uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bias_buf = match bias {
            Some(b) => b.buffer()?,
            None => input.buffer()?,
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.conv2d.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.conv2d);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                (w_out as f32 / 8.0).ceil() as u32,
                (h_out as f32 / 8.0).ceil() as u32,
                n * c_out,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(
            out_buffer,
            num_elements,
            4,
            self.pool.clone(),
        ))
    }

    fn conv2d_backward_input(
        &self,
        grad_out: &Self::Storage,
        weight: &Self::Storage,
        in_shape: &Shape,
        weight_shape: &Shape,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let n = in_shape[0] as u32;
        let c_in = in_shape[1] as u32;
        let h_in = in_shape[2] as u32;
        let w_in = in_shape[3] as u32;
        let c_out = weight_shape[0] as u32;
        let k_h = weight_shape[2] as u32;
        let k_w = weight_shape[3] as u32;
        let h_out =
            (h_in + 2 * padding as u32 - (dilation as u32 * (k_h - 1) + 1)) / stride as u32 + 1;
        let w_out =
            (w_in + 2 * padding as u32 - (dilation as u32 * (k_w - 1) + 1)) / stride as u32 + 1;

        let num_elements = in_shape.num_elements();
        let grad_in = self.allocate_buffer(
            (num_elements * 4) as wgpu::BufferAddress,
            wgpu::BufferUsages::STORAGE,
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_buffer(&grad_in, 0, None);

        let mut uniforms = [0u32; 16];
        uniforms[0] = n;
        uniforms[1] = c_in;
        uniforms[2] = h_in;
        uniforms[3] = w_in;
        uniforms[4] = c_out;
        uniforms[5] = k_h;
        uniforms[6] = k_w;
        uniforms[7] = h_out;
        uniforms[8] = w_out;
        uniforms[9] = padding as u32;
        uniforms[10] = stride as u32;
        uniforms[12] = dilation as u32;
        uniforms[13] = groups as u32;

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("conv2d_bw_uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.conv2d_bw_input.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_out.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.conv2d_bw_input);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                (w_out as f32 / 8.0).ceil() as u32,
                (h_out as f32 / 8.0).ceil() as u32,
                n * c_out,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(grad_in, num_elements, 4, self.pool.clone()))
    }

    fn conv2d_backward_weight(
        &self,
        grad_out: &Self::Storage,
        input: &Self::Storage,
        in_shape: &Shape,
        weight_shape: &Shape,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let n = in_shape[0] as u32;
        let c_in = in_shape[1] as u32;
        let h_in = in_shape[2] as u32;
        let w_in = in_shape[3] as u32;
        let c_out = weight_shape[0] as u32;
        let k_h = weight_shape[2] as u32;
        let k_w = weight_shape[3] as u32;
        let h_out = (in_shape[2] + 2 * padding - dilation * (weight_shape[2] - 1) - 1) / stride + 1;
        let w_out = (in_shape[3] + 2 * padding - dilation * (weight_shape[3] - 1) - 1) / stride + 1;

        let num_elements = weight_shape.num_elements();
        let grad_weight = self.allocate_buffer(
            (num_elements * 4) as wgpu::BufferAddress,
            wgpu::BufferUsages::STORAGE,
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_buffer(&grad_weight, 0, None);

        let mut uniforms = [0u32; 16];
        uniforms[0] = n;
        uniforms[1] = c_in;
        uniforms[2] = h_in;
        uniforms[3] = w_in;
        uniforms[4] = c_out;
        uniforms[5] = k_h;
        uniforms[6] = k_w;
        uniforms[7] = h_out as u32;
        uniforms[8] = w_out as u32;
        uniforms[9] = padding as u32;
        uniforms[10] = stride as u32;
        uniforms[12] = dilation as u32;
        uniforms[13] = groups as u32;

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("conv2d_bw_uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.conv2d_bw_weight.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_out.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_weight.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.conv2d_bw_weight);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                (w_out as f32 / 8.0).ceil() as u32,
                (h_out as f32 / 8.0).ceil() as u32,
                n * c_out,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(
            grad_weight,
            num_elements,
            4,
            self.pool.clone(),
        ))
    }

    fn conv2d_backward_bias(
        &self,
        grad_out: &Self::Storage,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let n = out_shape[0] as u32;
        let c_out = out_shape[1] as u32;
        let h_out = out_shape[2] as u32;
        let w_out = out_shape[3] as u32;

        let grad_bias = self.allocate_buffer(
            c_out as wgpu::BufferAddress * 4,
            wgpu::BufferUsages::STORAGE,
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_buffer(&grad_bias, 0, None);

        let mut uniforms = [0u32; 16];
        uniforms[0] = n;
        uniforms[4] = c_out;
        uniforms[7] = h_out;
        uniforms[8] = w_out;

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("conv2d_bw_uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.conv2d_bw_bias.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_out.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_out.buffer()?.as_entire_binding(),
                }, // dummy
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.conv2d_bw_bias);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                (w_out as f32 / 8.0).ceil() as u32,
                (h_out as f32 / 8.0).ceil() as u32,
                n * c_out,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(
            grad_bias,
            c_out as usize,
            4,
            self.pool.clone(),
        ))
    }

    fn mul_scalar(
        &self,
        storage: &Self::Storage,
        scalar: f32,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let num_elements = storage.num_elements;
        let out_buffer = self.allocate_buffer(
            (num_elements * 4) as wgpu::BufferAddress,
            wgpu::BufferUsages::STORAGE,
        );

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scalar_uniform"),
                contents: bytemuck::cast_slice(&[scalar]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group_layout = self.pipelines.mul_scalar.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.mul_scalar);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(((num_elements as f32) / 64.0).ceil() as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(
            out_buffer,
            num_elements,
            4,
            self.pool.clone(),
        ))
    }

    fn matmul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let lhs_rank = lhs_shape.rank();
        let rhs_rank = rhs_shape.rank();

        if lhs_rank < 2 || rhs_rank < 2 {
            return Err(OrcaError::InternalError(
                "Matmul requires at least 2D shapes".into(),
            ));
        }

        let m = lhs_shape[lhs_rank - 2] as u32;
        let k = lhs_shape[lhs_rank - 1] as u32;
        let n = rhs_shape[rhs_rank - 1] as u32;

        let mut batch_size = 1;
        for i in 0..lhs_rank - 2 {
            if lhs_shape[i] != rhs_shape[i] {
                return Err(OrcaError::InternalError(
                    "Matmul batch dimensions must match".into(),
                ));
            }
            batch_size *= lhs_shape[i];
        }
        let b = batch_size as u32;

        let num_elements = (b * m * n) as usize;
        let out_buffer = self.allocate_buffer(
            (num_elements * 4) as wgpu::BufferAddress,
            wgpu::BufferUsages::STORAGE,
        );

        // uniforms: M, K, N, B
        let uniforms: [u32; 4] = [m, k, n, b];
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matmul_uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group_layout = self.pipelines.matmul.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.matmul);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                (n as f32 / 16.0).ceil() as u32,
                (m as f32 / 16.0).ceil() as u32,
                b,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(
            out_buffer,
            num_elements,
            4,
            self.pool.clone(),
        ))
    }

    fn reshape(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        Ok(storage.clone()) // buffer memory doesn't change
    }

    fn transpose(
        &self,
        storage: &Self::Storage,
        shape: &Shape,
        dim0: usize,
        dim1: usize,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let rank = shape.rank();
        if dim0 >= rank || dim1 >= rank {
            return Err(OrcaError::InternalError(
                "Invalid transpose dimensions".into(),
            ));
        }

        let num_elements = shape.num_elements();
        let out_buffer = self.allocate_buffer(
            (num_elements * 4) as wgpu::BufferAddress,
            wgpu::BufferUsages::STORAGE,
        );

        let mut out_shape_vec = shape.to_vec();
        out_shape_vec.swap(dim0, dim1);
        let out_shape = Shape::new(out_shape_vec);

        // Pad shape to rank 4 to match transpose.wgsl expectation
        let padded_in = CpuBackend::pad_shape_left(&shape.0, 4);
        let in_strides = CpuBackend::compute_strides(&padded_in);
        let padded_out = CpuBackend::pad_shape_left(&out_shape.0, 4);
        let out_strides = CpuBackend::compute_strides(&padded_out);

        // Map dim0 and dim1 to their padded indices (left-padded to 4)
        let pad_offset = 4 - rank.min(4);
        let padded_dim0 = dim0 + pad_offset;
        let padded_dim1 = dim1 + pad_offset;

        let mut uniforms = [0u32; 16]; // Correct layout for the Uniforms struct in transpose.wgsl
        for i in 0..4 {
            uniforms[i] = in_strides[i] as u32; // in_strides
            uniforms[4 + i] = out_strides[i] as u32; // out_strides
            uniforms[8 + i] = padded_out[i] as u32; // out_shape
        }
        uniforms[12] = padded_dim0 as u32;
        uniforms[13] = padded_dim1 as u32;
        uniforms[14] = rank as u32;
        uniforms[15] = num_elements as u32;

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("transpose_uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group_layout = self.pipelines.transpose.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.transpose);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((num_elements as f32) / 256.0).ceil() as u32;
            let mut x = workgroups;
            let mut y = 1;
            if x > 65535 {
                y = workgroups.div_ceil(65535);
                x = 65535;
            }
            cpass.dispatch_workgroups(x, y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(
            out_buffer,
            num_elements,
            4,
            self.pool.clone(),
        ))
    }

    fn expand(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let num_elements = out_shape.num_elements();
        let out_buffer = self.allocate_buffer(
            (num_elements * 4) as wgpu::BufferAddress,
            wgpu::BufferUsages::STORAGE,
        );

        let rank = out_shape.rank();
        let padded_in = CpuBackend::pad_shape_left(&in_shape.0, rank);
        let in_strides = CpuBackend::compute_strides(&padded_in);
        let out_strides = CpuBackend::compute_strides(&out_shape.0);

        let mut uniforms = [0u32; 20];
        for i in 0..rank.min(4) {
            uniforms[i] = in_strides[i] as u32;
            uniforms[4 + i] = out_strides[i] as u32;
            uniforms[8 + i] = padded_in[i] as u32;
            uniforms[12 + i] = out_shape.0[i] as u32;
        }
        uniforms[16] = rank as u32;

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("expand_uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group_layout = self.pipelines.expand.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.expand);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(((num_elements as f32) / 64.0).ceil() as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(
            out_buffer,
            num_elements,
            4,
            self.pool.clone(),
        ))
    }

    fn sum_to_shape(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let num_elements = out_shape.num_elements();

        let out_buffer = self.allocate_buffer(
            (num_elements * 4) as wgpu::BufferAddress,
            wgpu::BufferUsages::STORAGE,
        );

        // Initialize output buffer with 0.0 (which is 0x00000000)
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_buffer(&out_buffer, 0, None);

        let rank = in_shape.rank();
        let padded_out = CpuBackend::pad_shape_left(&out_shape.0, rank);
        let in_strides = CpuBackend::compute_strides(&in_shape.0);
        let out_strides = CpuBackend::compute_strides(&padded_out);

        let mut uniforms = [0u32; 20];
        for i in 0..rank.min(4) {
            uniforms[i] = in_strides[i] as u32;
            uniforms[4 + i] = out_strides[i] as u32;
            uniforms[8 + i] = in_shape.0[i] as u32;
            uniforms[12 + i] = padded_out[i] as u32;
        }
        uniforms[16] = rank as u32;

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reduce_uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group_layout = self.pipelines.sum_to_shape.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.sum_to_shape);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(((storage.num_elements as f32) / 64.0).ceil() as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(
            out_buffer,
            num_elements,
            4,
            self.pool.clone(),
        ))
    }

    // Phase 6.1 Casting
    fn cast(
        &self,
        storage: &Self::Storage,
        _shape: &Shape,
        current_dtype: DType,
        target_dtype: DType,
    ) -> Result<Self::Storage> {
        if current_dtype == target_dtype {
            return Ok(storage.clone());
        }

        let supported = matches!(
            (current_dtype, target_dtype),
            (
                DType::F32 | DType::F16 | DType::BF16,
                DType::F32 | DType::F16 | DType::BF16
            )
        );
        if supported {
            Ok(storage.clone())
        } else {
            Err(OrcaError::UnsupportedDType {
                op: "cast (gpu)",
                dtype: target_dtype,
            })
        }
    }

    // Phase 2.1 Indexing
    fn scatter(
        &self,
        storage: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        src: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let num_elements = index_shape.num_elements();

        // Output starts as a copy of the base tensor (storage)
        let out_size = (shape.num_elements() * 4) as wgpu::BufferAddress;
        let out_buffer = self.allocate_buffer(out_size, wgpu::BufferUsages::STORAGE);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(storage.buffer()?, 0, &out_buffer, 0, out_size);

        let rank = shape.rank();
        let mut out_strides_pad = [0u32; 4];
        let mut idx_strides_pad = [0u32; 4];
        let out_strides = CpuBackend::compute_strides(&shape.0);
        let idx_strides = CpuBackend::compute_strides(&index_shape.0);
        for i in 0..rank {
            out_strides_pad[i] = out_strides[i] as u32;
            idx_strides_pad[i] = idx_strides[i] as u32;
        }

        let mut uniforms = [0u32; 12];
        uniforms[0] = dim as u32;
        uniforms[1] = rank as u32;
        uniforms[2] = num_elements as u32;
        uniforms[4] = out_strides_pad[0];
        uniforms[5] = out_strides_pad[1];
        uniforms[6] = out_strides_pad[2];
        uniforms[7] = out_strides_pad[3];
        uniforms[8] = idx_strides_pad[0];
        uniforms[9] = idx_strides_pad[1];
        uniforms[10] = idx_strides_pad[2];
        uniforms[11] = idx_strides_pad[3];

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scatter_uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.scatter.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: index.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.scatter);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(((num_elements as f32) / 64.0).ceil() as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(
            out_buffer,
            shape.num_elements(),
            4,
            self.pool.clone(),
        ))
    }

    fn gather(
        &self,
        storage: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let num_elements = index_shape.num_elements();
        let out_buffer = self.allocate_buffer(
            (num_elements * 4) as wgpu::BufferAddress,
            wgpu::BufferUsages::STORAGE,
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let rank = shape.rank();
        let mut in_strides_pad = [0u32; 4];
        let mut idx_strides_pad = [0u32; 4];
        let in_strides = CpuBackend::compute_strides(&shape.0);
        let idx_strides = CpuBackend::compute_strides(&index_shape.0);
        for i in 0..rank {
            in_strides_pad[i] = in_strides[i] as u32;
            idx_strides_pad[i] = idx_strides[i] as u32;
        }

        let mut uniforms = [0u32; 12];
        uniforms[0] = dim as u32;
        uniforms[1] = rank as u32;
        uniforms[2] = num_elements as u32;
        uniforms[4] = in_strides_pad[0];
        uniforms[5] = in_strides_pad[1];
        uniforms[6] = in_strides_pad[2];
        uniforms[7] = in_strides_pad[3];
        uniforms[8] = idx_strides_pad[0];
        uniforms[9] = idx_strides_pad[1];
        uniforms[10] = idx_strides_pad[2];
        uniforms[11] = idx_strides_pad[3];

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gather_uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.gather.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: index.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.gather);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(((num_elements as f32) / 64.0).ceil() as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(
            out_buffer,
            num_elements,
            4,
            self.pool.clone(),
        ))
    }

    fn scatter_backward_src(
        &self,
        grad_out: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        self.gather(grad_out, dim, index, shape, index_shape, dtype)
    }

    fn scatter_backward_base(
        &self,
        grad_out: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let num_elements = index_shape.num_elements();

        let out_size = (shape.num_elements() * 4) as wgpu::BufferAddress;
        let out_buffer = self.allocate_buffer(out_size, wgpu::BufferUsages::STORAGE);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // copy grad_out to out_buffer
        encoder.copy_buffer_to_buffer(grad_out.buffer()?, 0, &out_buffer, 0, out_size);

        let rank = shape.rank();
        let mut out_strides_pad = [0u32; 4];
        let mut idx_strides_pad = [0u32; 4];
        let out_strides = CpuBackend::compute_strides(&shape.0);
        let idx_strides = CpuBackend::compute_strides(&index_shape.0);
        for i in 0..rank {
            out_strides_pad[i] = out_strides[i] as u32;
            idx_strides_pad[i] = idx_strides[i] as u32;
        }

        let mut uniforms = [0u32; 12];
        uniforms[0] = dim as u32;
        uniforms[1] = rank as u32;
        uniforms[2] = num_elements as u32;
        uniforms[4] = out_strides_pad[0];
        uniforms[5] = out_strides_pad[1];
        uniforms[6] = out_strides_pad[2];
        uniforms[7] = out_strides_pad[3];
        uniforms[8] = idx_strides_pad[0];
        uniforms[9] = idx_strides_pad[1];
        uniforms[10] = idx_strides_pad[2];
        uniforms[11] = idx_strides_pad[3];

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scatter_bw_base_uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.scatter_bw_base.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: index.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.scatter_bw_base);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(((num_elements as f32) / 64.0).ceil() as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(
            out_buffer,
            shape.num_elements(),
            4,
            self.pool.clone(),
        ))
    }

    fn gather_backward(
        &self,
        grad_out: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let in_elements = shape.num_elements();
        let idx_dim_size = index_shape[dim];
        let out_size = (in_elements * 4) as wgpu::BufferAddress;
        let out_buffer = self.allocate_buffer(out_size, wgpu::BufferUsages::STORAGE);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // grad_in is initialized directly in the shader, but we can clear it just to be safe
        encoder.clear_buffer(&out_buffer, 0, None);

        let rank = shape.rank();
        let mut in_strides_pad = [0u32; 4];
        let mut idx_strides_pad = [0u32; 4];
        let in_strides = CpuBackend::compute_strides(&shape.0);
        let idx_strides = CpuBackend::compute_strides(&index_shape.0);

        for i in 0..rank {
            in_strides_pad[i] = in_strides[i] as u32;
            idx_strides_pad[i] = idx_strides[i] as u32;
        }

        let mut uniforms = [0u32; 12];
        uniforms[0] = dim as u32;
        uniforms[1] = rank as u32;
        uniforms[2] = in_elements as u32;
        uniforms[3] = idx_dim_size as u32;
        uniforms[4] = idx_strides_pad[0];
        uniforms[5] = idx_strides_pad[1];
        uniforms[6] = idx_strides_pad[2];
        uniforms[7] = idx_strides_pad[3];
        uniforms[8] = in_strides_pad[0];
        uniforms[9] = in_strides_pad[1];
        uniforms[10] = in_strides_pad[2];
        uniforms[11] = in_strides_pad[3];

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gather_bw_uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.gather_bw.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_out.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: index.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.gather_bw);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(((in_elements as f32) / 64.0).ceil() as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(
            out_buffer,
            shape.num_elements(),
            4,
            self.pool.clone(),
        ))
    }

    // Phase 1-2 Production Hardening
    fn from_bytes(&self, shape: &Shape, bytes: &[u8], dtype: DType) -> Result<Self::Storage> {
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("from_bytes"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        Ok(GpuStorage::new(
            buffer,
            shape.num_elements(),
            dtype.element_size(),
            self.pool.clone(),
        ))
    }

    fn to_bytes(&self, storage: &Self::Storage) -> Result<Vec<u8>> {
        let size = (storage.num_elements * storage.element_size) as wgpu::BufferAddress;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_bytes"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(storage.buffer()?, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });
        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = receiver.recv() {
            let data = slice.get_mapped_range();
            let result = data.to_vec();
            drop(data);
            staging.unmap();
            Ok(result)
        } else {
            Err(OrcaError::InternalError(
                "Failed to map GPU buffer for to_bytes".to_string(),
            ))
        }
    }

    fn has_nan_or_inf(&self, storage: &Self::Storage, _dtype: DType) -> Result<bool> {
        let num_elements = storage.num_elements;

        let out_buffer = self.allocate_buffer(4, wgpu::BufferUsages::STORAGE);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_buffer(&out_buffer, 0, None);

        let bind_group_layout = self.pipelines.check_nan_inf.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.check_nan_inf);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(((num_elements as f32) / 256.0).ceil() as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        // read back the 4 bytes using to_bytes trick
        let tmp_storage = GpuStorage::new(out_buffer, 1, 4, self.pool.clone());
        let bytes = self.to_bytes(&tmp_storage)?;
        let val = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

        Ok(val > 0)
    }

    fn max_to_shape(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let num_elements = out_shape.num_elements();

        let out_buffer = self.allocate_buffer(
            (num_elements * 4) as wgpu::BufferAddress,
            wgpu::BufferUsages::STORAGE,
        );

        let rank = in_shape.rank();
        let padded_out = CpuBackend::pad_shape_left(&out_shape.0, rank);
        let in_strides = CpuBackend::compute_strides(&in_shape.0);
        let out_strides = CpuBackend::compute_strides(&padded_out);

        let mut uniforms = [0u32; 20];
        for i in 0..rank.min(4) {
            uniforms[i] = in_strides[i] as u32;
            uniforms[4 + i] = out_strides[i] as u32;
            uniforms[8 + i] = in_shape.0[i] as u32;
            uniforms[12 + i] = padded_out[i] as u32;
        }
        uniforms[16] = rank as u32;

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("max_to_shape_uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group_layout = self.pipelines.max_to_shape.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.max_to_shape);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(((num_elements as f32) / 64.0).ceil() as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(
            out_buffer,
            num_elements,
            4,
            self.pool.clone(),
        ))
    }

    fn max_to_shape_backward(
        &self,
        grad_out: &Self::Storage,
        in_primal: &Self::Storage,
        out_primal: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let num_elements = in_shape.num_elements();

        let grad_in_buffer = self.allocate_buffer(
            (num_elements * 4) as wgpu::BufferAddress,
            wgpu::BufferUsages::STORAGE,
        );

        let rank = in_shape.rank();
        let padded_out = CpuBackend::pad_shape_left(&out_shape.0, rank);
        let in_strides = CpuBackend::compute_strides(&in_shape.0);
        let out_strides = CpuBackend::compute_strides(&padded_out);

        let mut uniforms = [0u32; 20];
        for i in 0..rank.min(4) {
            uniforms[i] = in_strides[i] as u32;
            uniforms[4 + i] = out_strides[i] as u32;
            uniforms[8 + i] = in_shape.0[i] as u32;
            uniforms[12 + i] = padded_out[i] as u32;
        }
        uniforms[16] = rank as u32;

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("max_to_shape_bw_uniforms"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group_layout = self.pipelines.max_to_shape_bw.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_out.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_primal.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_primal.buffer()?.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_in_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.max_to_shape_bw);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                ((out_shape.num_elements() as f32) / 64.0).ceil() as u32,
                1,
                1,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuStorage::new(
            grad_in_buffer,
            num_elements,
            4,
            self.pool.clone(),
        ))
    }
}
