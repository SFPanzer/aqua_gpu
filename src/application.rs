use crate::vulkan_context::VulkanContext;

pub(crate) struct Application {
    vulkan_context: VulkanContext,
}

impl Application {
    pub fn initialize(name: &str, width: u32, height: u32) -> Self {
        let vulkan_context = VulkanContext::initialize(name, width, height);
        Self { vulkan_context }
    }

    pub fn main_loop(&mut self){
        self.vulkan_context.main_loop();
    }
}
