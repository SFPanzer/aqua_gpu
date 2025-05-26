use std::sync::Arc;

use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        BufferUsage,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo,
    },
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
            DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo,
        },
        Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions,
    },
    memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator},
    VulkanLibrary,
};

use super::{traits::GpuTaskExecutor, GpuTask};

pub(crate) struct VulkanoHeadlessBackend {
    instance: Arc<Instance>,
    _debug_messenger: Option<DebugUtilsMessenger>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    uniform_buffer_allocator: SubbufferAllocator,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl VulkanoHeadlessBackend {
    pub fn new() -> Self {
        let instance = get_vulkan_instance();
        let _debug_messenger = get_debug_messenger(&instance);
        let (device, queue) = get_device_and_queue(&instance);
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let uniform_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        Self {
            instance,
            _debug_messenger,
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            uniform_buffer_allocator,
            descriptor_set_allocator,
        }
    }

    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    #[allow(unused)]
    pub fn uniform_buffer_allocator(&self) -> &SubbufferAllocator {
        &self.uniform_buffer_allocator
    }

    pub fn command_buffer_builder(&self) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
    }

    pub fn descriptor_set_allocator(&self) -> &Arc<StandardDescriptorSetAllocator> {
        &self.descriptor_set_allocator
    }

    pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator> {
        &self.memory_allocator
    }
}

impl GpuTaskExecutor for VulkanoHeadlessBackend {
    fn execute(&self, task: &mut dyn GpuTask) {
        let mut builder = self.command_buffer_builder();
        task.record(&mut builder);
        let command_buffer = builder.build().unwrap();
        task.submit(command_buffer, &self.queue, &self.device);
    }
}

fn get_vulkan_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().unwrap();
    let extensions = InstanceExtensions {
        ext_debug_utils: true,
        ..InstanceExtensions::empty()
    };
    let layers = vec!["VK_LAYER_KHRONOS_validation".to_owned()];

    Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_layers: layers,
            enabled_extensions: extensions,
            ..Default::default()
        },
    )
    .expect("failed to create Vulkan instance")
}

fn get_debug_messenger(instance: &Arc<Instance>) -> Option<DebugUtilsMessenger> {
    unsafe {
        DebugUtilsMessenger::new(
            instance.clone(),
            DebugUtilsMessengerCreateInfo {
                message_severity: DebugUtilsMessageSeverity::ERROR
                    | DebugUtilsMessageSeverity::WARNING,
                message_type: DebugUtilsMessageType::GENERAL
                    | DebugUtilsMessageType::VALIDATION
                    | DebugUtilsMessageType::PERFORMANCE,
                ..DebugUtilsMessengerCreateInfo::user_callback(DebugUtilsMessengerCallback::new(
                    |message_severity, message_type, callback_data| {
                        let severity = if message_severity
                            .intersects(DebugUtilsMessageSeverity::ERROR)
                        {
                            "error"
                        } else if message_severity.intersects(DebugUtilsMessageSeverity::WARNING) {
                            "warning"
                        } else if message_severity.intersects(DebugUtilsMessageSeverity::INFO) {
                            "information"
                        } else if message_severity.intersects(DebugUtilsMessageSeverity::VERBOSE) {
                            "verbose"
                        } else {
                            panic!("no-impl");
                        };

                        let ty = if message_type.intersects(DebugUtilsMessageType::GENERAL) {
                            "general"
                        } else if message_type.intersects(DebugUtilsMessageType::VALIDATION) {
                            "validation"
                        } else if message_type.intersects(DebugUtilsMessageType::PERFORMANCE) {
                            "performance"
                        } else {
                            panic!("no-impl");
                        };

                        println!(
                            "{} {} {}: {}",
                            callback_data.message_id_name.unwrap_or("unknown"),
                            ty,
                            severity,
                            callback_data.message
                        );
                    },
                ))
            },
        )
    }
    .ok()
}

fn get_device_and_queue(instance: &Arc<Instance>) -> (Arc<Device>, Arc<Queue>) {
    let device_extensions = DeviceExtensions {
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .map(|p| {
            (!p.queue_family_properties().is_empty())
                .then_some((p, 0))
                .expect("couldn't find a queue family")
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("no device available");

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("failed to create device");
    let queue = queues.next().unwrap();

    (device, queue)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_headless_backend_creation() {
        let backend = VulkanoHeadlessBackend::new();

        assert!(Arc::strong_count(backend.instance()) > 0);
        assert!(Arc::strong_count(backend.device()) > 0);
        assert!(Arc::strong_count(backend.memory_allocator()) > 0);
        assert!(Arc::strong_count(backend.descriptor_set_allocator()) > 0);
        assert!(Arc::strong_count(&backend.command_buffer_allocator) > 0);
    }
}
