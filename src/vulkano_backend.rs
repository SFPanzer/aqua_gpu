use std::sync::Arc;

use vulkano::{
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    VulkanLibrary,
};
use vulkano_taskgraph::{
    graph::ExecutableTaskGraph,
    resource::{Flight, Resources},
    Id,
};
use winit::event_loop::EventLoop;

const MAX_FRAMES_IN_FLIGHT: u32 = 2;
const MAX_NODES_IN_TASK_GRAPH: u32 = 64;
const MAX_RESOURCES_IN_TASK_GRAPH: u32 = 64;

pub(crate) struct VulkanoBackend {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,
    task_graph: Option<ExecutableTaskGraph<Self>>,
}

impl VulkanoBackend {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let instance = get_vulkan_instance(event_loop);
        let (device, queue) = get_device_and_queue(&instance, event_loop);

        let resources = Resources::new(&device, &Default::default());
        let flight_id = resources.create_flight(MAX_FRAMES_IN_FLIGHT).unwrap();

        Self {
            instance,
            device,
            queue,
            resources,
            flight_id,
            task_graph: None,
        }
    }

    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn create_swapchain(&self, surface: &Arc<Surface>, create_info: SwapchainCreateInfo) -> Id<Swapchain> {
        self.resources
            .create_swapchain(self.flight_id, surface.clone(), create_info).unwrap()
    }
}

fn get_vulkan_instance(event_loop: &EventLoop<()>) -> Arc<Instance> {
    let required_extensions = Surface::required_extensions(event_loop).unwrap();

    let library = VulkanLibrary::new().unwrap();
    Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap()
}

fn get_device_and_queue(
    instance: &Arc<Instance>,
    event_loop: &EventLoop<()>,
) -> (Arc<Device>, Arc<Queue>) {
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.presentation_support(i as u32, event_loop).unwrap()
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| {
            // We assign a lower score to device types that are likely to be faster/better.
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .expect("no suitable physical device found");

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
    .unwrap();
    let queue = queues.next().unwrap();

    (device, queue)
}
