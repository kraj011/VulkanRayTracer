use std::sync::Arc;

use vulkano::{
    Packed24_8, Validated, VulkanError, VulkanLibrary,
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureCreateInfo,
        AccelerationStructureGeometries, AccelerationStructureGeometryInstancesData,
        AccelerationStructureInstance, BuildAccelerationStructureFlags, GeometryInstanceFlags,
    },
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyImageInfo, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDevice,
    },
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
        StandardMemoryAllocator,
    },
    pipeline::{
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        ray_tracing::{
            RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
            ShaderBindingTable,
        },
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{self, GpuFuture, future::FenceSignalFuture},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use crate::{
    material::EngineMaterial,
    parser::Parser,
    shaders::{self, closest_hit_shader, miss_shader, ray_gen_shader},
    state::State,
    vertex::EngineVertex,
};

#[repr(C)]
#[derive(Clone, Copy, BufferContents)]
pub struct GeometryBindings {
    pub vertex_addr: u64,
    pub index_addr: u64,
    pub material_idx: u32,
    pub _pad: u32,
}

pub struct Engine {
    pub state: State,

    device: Arc<Device>,
    physical_device: Arc<PhysicalDevice>,
    queue: Arc<Queue>,
    queue_family_index: u32,
    memory_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    window: Arc<Window>,
    surface: Arc<Surface>,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,

    pipeline: Arc<GraphicsPipeline>,
    framebuffers: Vec<Arc<Framebuffer>>,

    rt_pipeline: Arc<RayTracingPipeline>,
    sbt: Arc<ShaderBindingTable>,
    rt_storage_image: Arc<Image>,
    rt_storage_image_view: Arc<ImageView>,
}

impl Engine {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new()
            .expect("Failed to initialize library. No local Vulkan library or DLL.");
        let required_exts = Surface::required_extensions(&event_loop).unwrap();
        let instance = Instance::new(
            library.clone(),
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_exts,
                ..Default::default()
            },
        )
        .expect("Failed to create instance.");

        let physical_device = instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices.")
            .next()
            .expect("No devices available.");

        let device_name = physical_device.properties().device_name.clone();

        println!("Attaching to {}!", device_name);

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .position(|properties| properties.queue_flags.contains(QueueFlags::GRAPHICS))
            .expect("Couldn't find a graphics queue family, ensure GPU has support for it.")
            as u32;

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_ray_tracing_pipeline: true,
            khr_acceleration_structure: true,
            khr_deferred_host_operations: true,
            ..DeviceExtensions::empty()
        };
        let enabled_features = DeviceFeatures {
            acceleration_structure: true,
            buffer_device_address: true,
            ray_tracing_pipeline: true,
            shader_int64: true,
            ..Default::default()
        };
        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: enabled_features,
                ..Default::default()
            },
        )
        .expect("Failed to create device.");

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        // WINDOW SETUP
        let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
        window.set_title("David's Renderer");
        window.set_resizable(false);
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        let dimensions = window.inner_size();
        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = physical_device
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count + 1,
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                composite_alpha,
                ..Default::default()
            },
        )
        .unwrap();

        // TODO: Move chunks of setup to their own functions
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap();

        let framebuffers = images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let vs = shaders::preiew_vertex_shader::load(device.clone())
            .expect("Failed to create vertex shader.");
        let fs = shaders::preiew_fragment_shader::load(device.clone())
            .expect("Failed to create fragment shader.");

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        let pipeline = Engine::get_pipeline(
            device.clone(),
            vs.clone(),
            fs.clone(),
            render_pass.clone(),
            viewport.clone(),
        );

        let state = State {
            recreate_swapchain: false,
            frame_count: 1,
        };

        let (rt_pipeline, sbt) =
            Engine::create_ray_trace_pipeline(device.clone(), memory_allocator.clone()).unwrap();

        let rt_storage_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: swapchain.image_format(),
                extent: [dimensions.width, dimensions.height, 1],
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        // Transition storage image from Undefined to General layout (required for ray tracing)
        {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder
                .clear_color_image(vulkano::command_buffer::ClearColorImageInfo::image(
                    rt_storage_image.clone(),
                ))
                .unwrap();

            let command_buffer = builder.build().unwrap();

            sync::now(device.clone())
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();
        }

        let rt_storage_image_view = ImageView::new_default(rt_storage_image.clone()).unwrap();

        Engine {
            state,
            device,
            physical_device,
            queue,
            queue_family_index,
            memory_allocator,
            command_buffer_allocator,

            window,
            surface,
            swapchain,
            images,

            pipeline,
            framebuffers,

            rt_pipeline,
            sbt,
            rt_storage_image,
            rt_storage_image_view,
        }
    }

    fn get_pipeline(
        device: Arc<Device>,
        vs: Arc<ShaderModule>,
        fs: Arc<ShaderModule>,
        render_pass: Arc<RenderPass>,
        viewport: Viewport,
    ) -> Arc<GraphicsPipeline> {
        let vs = vs.entry_point("main").unwrap();
        let fs = fs.entry_point("main").unwrap();

        let vertex_input_state = EngineVertex::per_vertex().definition(&vs).unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [viewport].into_iter().collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::None, // Disable backface culling
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    }

    fn get_command_buffers(
        &self,
        queue: &Arc<Queue>,
        pipeline: &Arc<GraphicsPipeline>,
        framebuffers: &Vec<Arc<Framebuffer>>,
        parser: &Parser,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
        framebuffers
            .iter()
            .map(|framebuffer| {
                let mut builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> =
                    AutoCommandBufferBuilder::primary(
                        self.command_buffer_allocator.clone(),
                        queue.queue_family_index(),
                        CommandBufferUsage::MultipleSubmit,
                    )
                    .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap();

                parser.meshes.iter().for_each(|mesh| {
                    if mesh.vertex_buffer.is_none() || mesh.index_buffer.is_none() {
                        return;
                    }

                    let push_constants = shaders::preiew_vertex_shader::PushConstants {
                        mvp: mesh.mvp.to_cols_array_2d(),
                        col: [rand::random(), rand::random(), rand::random()],
                    };

                    unsafe {
                        builder
                            .bind_vertex_buffers(0, mesh.vertex_buffer.as_ref().unwrap().clone())
                            .unwrap()
                            .bind_index_buffer(mesh.index_buffer.as_ref().unwrap().clone())
                            .unwrap()
                            .push_constants(pipeline.layout().clone(), 0, push_constants)
                            .unwrap()
                            .draw_indexed(mesh.indices.len() as u32, 1, 0, 0, 0)
                            .unwrap()
                    };
                });

                builder.end_render_pass(SubpassEndInfo::default()).unwrap();

                builder.build().unwrap()
            })
            .collect()
    }

    pub fn create_buffers(&self, parser: &mut Parser) {
        if parser.cameras.len() == 0 {
            panic!("No camera found. Please re-run with a camera.");
        }

        for mesh in parser.meshes.iter_mut() {
            if !mesh.vertex_buffer.is_none() {
                continue;
            }

            let camera = &parser.cameras[0];

            let vertex_buffer = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER
                        | BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                mesh.vertices.clone(), // TODO: Is this fine? Maybe free the mesh vertices after?
            );

            match vertex_buffer {
                Ok(buffer) => mesh.vertex_buffer = Some(buffer),
                Err(err) => panic!("Error creating vertex buffer for mesh {}", err),
            }

            let required_elements = mesh.indices.len() * 4 + 1; // This is actually bytes, but Vulkano checks .len()
            let mut padded_indices = mesh.indices.clone();
            while padded_indices.len() < required_elements {
                padded_indices.push(0);
            }

            let index_buffer = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER
                        | BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                padded_indices,
            );

            match index_buffer {
                Ok(buffer) => mesh.index_buffer = Some(vulkano::buffer::IndexBuffer::U32(buffer)),
                Err(err) => panic!("Error creating index buffer for mesh {}", err),
            }

            let correction = glam::Mat4::from_cols_array(&[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
            ]);

            mesh.mvp = correction * camera.get_projection() * camera.xform.inverse() * mesh.xform;
            // mesh.mvp = mesh.mvp.transpose();
        }
    }

    fn create_acceleration_structures(
        &self,
        geometries: AccelerationStructureGeometries,
        flags: BuildAccelerationStructureFlags,
        build_range_info: &AccelerationStructureBuildRangeInfo,
    ) -> Result<Arc<AccelerationStructure>, anyhow::Error> {
        // https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/#understanding-acceleration-structures-blas-vs-tlas
        let mut build_info = AccelerationStructureBuildGeometryInfo::new(geometries);
        build_info.flags = flags;
        build_info.mode = vulkano::acceleration_structure::BuildAccelerationStructureMode::Build;
        build_info.dst_acceleration_structure = None;

        let build_size = self.device.acceleration_structure_build_sizes(
            vulkano::acceleration_structure::AccelerationStructureBuildType::Device,
            &build_info,
            &[build_range_info.primitive_count],
        )?;

        let as_buffer = Buffer::new_slice::<u8>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE
                    | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            build_size.acceleration_structure_size,
        )?;

        let acceleration_structure = unsafe {
            AccelerationStructure::new(
                self.device.clone(),
                AccelerationStructureCreateInfo::new(as_buffer),
            )
        }?;

        let properties = self.physical_device.properties();
        let min_scratch_offset_alignment = properties
            .min_acceleration_structure_scratch_offset_alignment
            .unwrap() as u64;

        let scratch_size = (build_size.build_scratch_size + min_scratch_offset_alignment - 1)
            & !(min_scratch_offset_alignment - 1);

        let scratch_buffer = Buffer::new_slice::<u8>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            scratch_size,
        )?;

        build_info.dst_acceleration_structure = Some(acceleration_structure.clone());
        build_info.scratch_data = Some(scratch_buffer);

        // TODO: Create actual acceleration structure
        // TODO: Optimize. See bottom of section 2.2 for info

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        unsafe {
            builder
                .build_acceleration_structure(build_info, vec![*build_range_info].into())
                .unwrap()
        };

        let command_buffer = builder.build().unwrap();

        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        Ok(acceleration_structure)
    }

    fn create_blas(
        &self,
        parser: &Parser,
    ) -> Result<Vec<Arc<AccelerationStructure>>, anyhow::Error> {
        // NOTE: DO NOT USE TRANSFORMS HERE SINCE WE'LL TRANSFORM IT IN THE TLAS
        let mut blas_vec: Vec<Arc<AccelerationStructure>> = Vec::new();
        for mesh in &parser.meshes {
            let (range, geometry) = mesh.get_acceleration_structure();

            let blas = self.create_acceleration_structures(
                geometry,
                BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
                &range,
            )?;
            blas_vec.push(blas);
        }

        Ok(blas_vec)
    }

    fn create_tlas(
        &self,
        parser: &Parser,
        blas_vec: Vec<Arc<AccelerationStructure>>,
    ) -> Result<Arc<AccelerationStructure>, anyhow::Error> {
        let mut tlas_instances = Vec::new();
        for (i, mesh) in parser.meshes.iter().enumerate() {
            let xform = mesh.xform;
            let transform = [
                [
                    xform.x_axis[0],
                    xform.y_axis[0],
                    xform.z_axis[0],
                    xform.w_axis[0],
                ],
                [
                    xform.x_axis[1],
                    xform.y_axis[1],
                    xform.z_axis[1],
                    xform.w_axis[1],
                ],
                [
                    xform.x_axis[2],
                    xform.y_axis[2],
                    xform.z_axis[2],
                    xform.w_axis[2],
                ],
            ];
            let instance = AccelerationStructureInstance {
                transform: transform,
                instance_custom_index_and_mask: Packed24_8::new(i as u32, 0xFF),
                instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                    0,
                    GeometryInstanceFlags::TRIANGLE_FACING_CULL_DISABLE.into(),
                ),
                acceleration_structure_reference: blas_vec[i].device_address().get(),
            };

            tlas_instances.push(instance);
        }

        let tlas_instances_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                    | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            tlas_instances,
        )?;

        let geometry = AccelerationStructureGeometries::Instances(
            AccelerationStructureGeometryInstancesData::new(
                vulkano::acceleration_structure::AccelerationStructureGeometryInstancesDataType::Values(Some(tlas_instances_buffer)))
        );

        let build_range_info = AccelerationStructureBuildRangeInfo {
            primitive_count: parser.meshes.len() as u32,
            ..Default::default()
        };

        let tlas = self.create_acceleration_structures(
            geometry,
            BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
            &build_range_info,
        )?;

        Ok(tlas)
    }

    fn create_rt_descriptor_set(
        &self,
        tlas: Arc<AccelerationStructure>,
        geometry_bindings_buffer: Subbuffer<[GeometryBindings]>,
        materials_buffer: Subbuffer<[EngineMaterial]>,
    ) -> Arc<DescriptorSet> {
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            self.device.clone(),
            Default::default(),
        ));

        DescriptorSet::new(
            descriptor_set_allocator.clone(),
            self.rt_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::acceleration_structure(0, tlas.clone()),
                WriteDescriptorSet::image_view(1, self.rt_storage_image_view.clone()),
                WriteDescriptorSet::buffer(2, geometry_bindings_buffer),
                WriteDescriptorSet::buffer(3, materials_buffer),
            ],
            [],
        )
        .unwrap()
    }

    fn create_ray_trace_pipeline(
        device: Arc<Device>,
        memory_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    ) -> Result<(Arc<RayTracingPipeline>, Arc<ShaderBindingTable>), anyhow::Error> {
        let raygen_shader = ray_gen_shader::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let miss_shader = miss_shader::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let closest_hit_shader = closest_hit_shader::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let stages: Vec<_> = [
            PipelineShaderStageCreateInfo::new(raygen_shader),
            PipelineShaderStageCreateInfo::new(miss_shader),
            PipelineShaderStageCreateInfo::new(closest_hit_shader),
        ]
        .into_iter()
        .collect();

        let groups: Vec<_> = [
            // Raygen group (index 0 in stages)
            RayTracingShaderGroupCreateInfo::General { general_shader: 0 },
            // Miss group (index 1 in stages)
            RayTracingShaderGroupCreateInfo::General { general_shader: 1 },
            // Hit group (index 2 in stages - closest hit)
            RayTracingShaderGroupCreateInfo::TrianglesHit {
                closest_hit_shader: Some(2),
                any_hit_shader: None,
            },
        ]
        .into_iter()
        .collect();

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let pipeline_info = RayTracingPipelineCreateInfo {
            stages: stages.into(),
            groups: groups.into(),
            max_pipeline_ray_recursion_depth: 1,
            ..RayTracingPipelineCreateInfo::layout(layout)
        };

        let rt_pipeline = RayTracingPipeline::new(device.clone(), None, pipeline_info)?;

        let sbt = ShaderBindingTable::new(memory_allocator.clone(), rt_pipeline.as_ref())?;

        Ok((rt_pipeline, sbt.into()))
    }

    fn get_raytracing_command_buffers(
        &self,
        parser: &Parser,
        descriptor_set: &Arc<DescriptorSet>,
        swapchain_image: &Arc<Image>,
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let mut builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> =
            AutoCommandBufferBuilder::primary(
                self.command_buffer_allocator.clone(),
                self.queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

        unsafe {
            builder
                .bind_pipeline_ray_tracing(self.rt_pipeline.clone())
                .unwrap()
                .push_constants(
                    self.rt_pipeline.layout().clone(),
                    0,
                    ray_gen_shader::PushConstants {
                        projInv: parser.cameras[0]
                            .get_projection()
                            .inverse()
                            .to_cols_array_2d(),
                        viewInv: parser.cameras[0].xform.to_cols_array_2d(),
                        frameCount: self.state.frame_count,
                    },
                )
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::RayTracing,
                    self.rt_pipeline.layout().clone(),
                    0,
                    descriptor_set.clone(),
                )
                .unwrap()
                .trace_rays(
                    self.sbt.addresses().clone(),
                    [
                        self.window.inner_size().width,
                        self.window.inner_size().height,
                        1,
                    ],
                )
                .unwrap()
                .copy_image(CopyImageInfo::images(
                    self.rt_storage_image.clone(),
                    swapchain_image.clone(),
                ))
                .unwrap();
        }

        builder.build().unwrap()
    }

    fn create_geometry_bindings_buffer(&self, parser: &Parser) -> Subbuffer<[GeometryBindings]> {
        let bindings: Vec<GeometryBindings> = parser
            .meshes
            .iter()
            .map(|mesh| {
                let vertex_addr = mesh
                    .vertex_buffer
                    .as_ref()
                    .map(|b| b.buffer().device_address().unwrap().get())
                    .unwrap_or(0);

                let index_addr = match &mesh.index_buffer {
                    Some(IndexBuffer::U32(b)) => b.buffer().device_address().unwrap().get(),
                    _ => 0,
                };

                GeometryBindings {
                    vertex_addr,
                    index_addr,
                    material_idx: match &mesh.material_path {
                        Some(path) => *parser.material_map.get(path).unwrap() as u32,
                        None => 0, // TODO find a better default val
                    },
                    _pad: 0,
                }
            })
            .collect();

        Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            bindings,
        )
        .unwrap()
    }

    pub fn create_materials_buffer(&self, parser: &Parser) -> Subbuffer<[EngineMaterial]> {
        let materials: Vec<EngineMaterial> = parser
            .materials
            .iter()
            .map(|material| material.engine_material)
            .collect();

        Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            materials,
        )
        .unwrap()
    }

    pub fn run_rasterization(self, event_loop: EventLoop<()>, parser: &Parser) {
        let frames_in_flight = self.images.len();
        let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
        let mut previous_fence_i = 0;

        let command_buffers =
            self.get_command_buffers(&self.queue, &self.pipeline, &self.framebuffers, parser);

        let blas = self.create_blas(parser).unwrap();
        let tlas = self.create_tlas(parser, blas.clone()).unwrap();

        unsafe {
            self.device.wait_idle().unwrap();
        }

        let window = self.window.clone();
        let _ = event_loop.run(move |event, elwt| {
            // Keep BLAS and TLAS alive for the lifetime of the app
            let _ = (&blas, &tlas);

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    elwt.exit();
                }
                Event::AboutToWait => {
                    // Explicit redraw request creates clear frame boundaries for Nsight
                    window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    let (image_i, _suboptimal, acquire_future) =
                        match swapchain::acquire_next_image(self.swapchain.clone(), None)
                            .map_err(Validated::unwrap)
                        {
                            Ok(r) => r,
                            Err(VulkanError::OutOfDate) => {
                                // Do something here
                                // recereate swapchain
                                return;
                            }
                            Err(e) => panic!("Failed to acquire next image! {e}"),
                        };

                    if let Some(image_fence) = &fences[image_i as usize] {
                        image_fence.wait(None).unwrap();
                    }

                    let previous_future = match fences[previous_fence_i as usize].clone() {
                        None => {
                            let mut now = sync::now(self.device.clone());
                            now.cleanup_finished();
                            now.boxed()
                        }
                        Some(fence) => fence.boxed(),
                    };

                    let future = previous_future
                        .join(acquire_future)
                        .then_execute(
                            self.queue.clone(),
                            command_buffers[image_i as usize].clone(),
                        )
                        .unwrap()
                        .then_swapchain_present(
                            self.queue.clone(),
                            SwapchainPresentInfo::swapchain_image_index(
                                self.swapchain.clone(),
                                image_i,
                            ),
                        )
                        .then_signal_fence_and_flush();

                    fences[image_i as usize] = match future.map_err(Validated::unwrap) {
                        Ok(value) => Some(Arc::new(value)),
                        Err(VulkanError::OutOfDate) => {
                            // recreate swapchain
                            None
                        }
                        Err(e) => {
                            println!("Failed to flush future: {e}");
                            None
                        }
                    };

                    previous_fence_i = image_i;
                }
                _ => (),
            }
        });
    }

    pub fn run_rt(mut self, event_loop: EventLoop<()>, parser: &Parser) {
        let frames_in_flight = self.images.len();
        let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
        let mut previous_fence_i = 0;

        let blas = self.create_blas(parser).unwrap();
        let tlas = self.create_tlas(parser, blas.clone()).unwrap();

        let geometry_bindings_buffer = self.create_geometry_bindings_buffer(parser);
        let materials_buffer = self.create_materials_buffer(parser);

        let descriptor_set =
            self.create_rt_descriptor_set(tlas, geometry_bindings_buffer, materials_buffer);

        unsafe {
            self.device.wait_idle().unwrap();
        }

        let window = self.window.clone();
        let _ = event_loop.run(move |event, elwt| {
            let command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>> = self
                .images
                .iter()
                .map(|swapchain_image| {
                    self.get_raytracing_command_buffers(
                        parser,
                        &descriptor_set.clone(),
                        swapchain_image,
                    )
                })
                .collect();
            let _ = &blas;

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    elwt.exit();
                }
                Event::AboutToWait => {
                    // Explicit redraw request creates clear frame boundaries for Nsight
                    window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    self.state.frame_count += 1;

                    let (image_i, _suboptimal, acquire_future) =
                        match swapchain::acquire_next_image(self.swapchain.clone(), None)
                            .map_err(Validated::unwrap)
                        {
                            Ok(r) => r,
                            Err(VulkanError::OutOfDate) => {
                                // Do something here
                                // recereate swapchain
                                return;
                            }
                            Err(e) => panic!("Failed to acquire next image! {e}"),
                        };

                    if let Some(image_fence) = &fences[image_i as usize] {
                        image_fence.wait(None).unwrap();
                    }

                    let previous_future = match fences[previous_fence_i as usize].clone() {
                        None => {
                            let mut now = sync::now(self.device.clone());
                            now.cleanup_finished();
                            now.boxed()
                        }
                        Some(fence) => fence.boxed(),
                    };

                    let future = previous_future
                        .join(acquire_future)
                        .then_execute(
                            self.queue.clone(),
                            command_buffers[image_i as usize].clone(),
                        )
                        .unwrap()
                        .then_swapchain_present(
                            self.queue.clone(),
                            SwapchainPresentInfo::swapchain_image_index(
                                self.swapchain.clone(),
                                image_i,
                            ),
                        )
                        .then_signal_fence_and_flush();

                    fences[image_i as usize] = match future.map_err(Validated::unwrap) {
                        Ok(value) => Some(Arc::new(value)),
                        Err(VulkanError::OutOfDate) => {
                            // recreate swapchain
                            None
                        }
                        Err(e) => {
                            println!("Failed to flush future: {e}");
                            None
                        }
                    };

                    previous_fence_i = image_i;
                }
                _ => (),
            }
        });
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } buf;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                buf.data[idx] *= 12;
            }
        "
    }
}
