/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flutter.h"

#include <locale>

#include "common/logging.h"
#include "common/vk_common.h"
#include "glsl_compiler.h"
#include "platform/filesystem.h"
#include "platform/platform.h"

#include "channel/isolate.h"
#include "channel/navigation.h"
#include "channel/platform.h"
#include "channel/restoration.h"

namespace flutter
{

// Note: the default dispatcher is instantiated in hpp_api_vulkan_sample.cpp.
//			 Even though, that file is not part of this sample, it's part of the sample-project!

#if defined(VKB_DEBUG) || defined(VKB_VALIDATION_LAYERS)
/// @brief A debug callback called from Vulkan validation layers.
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT type,
                                                     uint64_t object, size_t location, int32_t message_code,
                                                     const char *layer_prefix, const char *message, void *user_data)
{
	if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
	{
		LOGE("Validation Layer: Error: {}: {}", layer_prefix, message);
	}
	else if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT)
	{
		LOGE("Validation Layer: Warning: {}: {}", layer_prefix, message);
	}
	else if (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)
	{
		LOGI("Validation Layer: Performance warning: {}: {}", layer_prefix, message);
	}
	else
	{
		LOGI("Validation Layer: Information: {}: {}", layer_prefix, message);
	}
	return VK_FALSE;
}
#endif

/**
 * @brief Validates a list of required extensions, comparing it with the available ones.
 *
 * @param required A vector containing required extension names.
 * @param available A vk::ExtensionProperties object containing available extensions.
 * @return true if all required extensions are available
 * @return false otherwise
 */
bool flutter::validate_extensions(const std::vector<const char *>            &required,
                                  const std::vector<vk::ExtensionProperties> &available)
{
	// inner find_if gives true if the extension was not found
	// outer find_if gives true if none of the extensions were not found, that is if all extensions were found
	return std::find_if(required.begin(),
	                    required.end(),
	                    [&available](auto extension) {
		                    return std::find_if(available.begin(),
		                                        available.end(),
		                                        [&extension](auto const &ep) {
			                                        return strcmp(ep.extensionName, extension) == 0;
		                                        }) == available.end();
	                    }) == required.end();
}

/**
 * @brief Find the vulkan shader stage for a given a string.
 *
 * @param ext A string containing the shader stage name.
 * @return vk::ShaderStageFlagBits The shader stage mapping from the given string, vk::ShaderStageFlagBits::eVertex otherwise.
 */
VkShaderStageFlagBits flutter::find_shader_stage(const std::string &ext)
{
	if (ext == "vert")
	{
		return VK_SHADER_STAGE_VERTEX_BIT;
	}
	else if (ext == "frag")
	{
		return VK_SHADER_STAGE_FRAGMENT_BIT;
	}
	else if (ext == "comp")
	{
		return VK_SHADER_STAGE_COMPUTE_BIT;
	}
	else if (ext == "geom")
	{
		return VK_SHADER_STAGE_GEOMETRY_BIT;
	}
	else if (ext == "tesc")
	{
		return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
	}
	else if (ext == "tese")
	{
		return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
	}

	throw std::runtime_error("No Vulkan shader stage found for the file extension name.");
};

bool validate_layers(const std::vector<const char *>        &required,
                     const std::vector<vk::LayerProperties> &available)
{
	// inner find_if returns true if the layer was not found
	// outer find_if returns iterator to the not found layer, if any
	auto requiredButNotFoundIt = std::find_if(required.begin(),
	                                          required.end(),
	                                          [&available](auto layer) {
		                                          return std::find_if(available.begin(),
		                                                              available.end(),
		                                                              [&layer](auto const &lp) {
			                                                              return strcmp(lp.layerName, layer) == 0;
		                                                              }) == available.end();
	                                          });
	if (requiredButNotFoundIt != required.end())
	{
		LOGE("Validation Layer {} not found", *requiredButNotFoundIt);
	}
	return (requiredButNotFoundIt == required.end());
}

std::vector<const char *> get_optimal_validation_layers(const std::vector<vk::LayerProperties> &supported_instance_layers)
{
	std::vector<std::vector<const char *>> validation_layer_priority_list =
	    {
	        // The preferred validation layer is "VK_LAYER_KHRONOS_validation"
	        {"VK_LAYER_KHRONOS_validation"},

	        // Otherwise we fallback to using the LunarG meta layer
	        {"VK_LAYER_LUNARG_standard_validation"},

	        // Otherwise we attempt to enable the individual layers that compose the LunarG meta layer since it doesn't exist
	        {
	            "VK_LAYER_GOOGLE_threading",
	            "VK_LAYER_LUNARG_parameter_validation",
	            "VK_LAYER_LUNARG_object_tracker",
	            "VK_LAYER_LUNARG_core_validation",
	            "VK_LAYER_GOOGLE_unique_objects",
	        },

	        // Otherwise as a last resort we fallback to attempting to enable the LunarG core layer
	        {"VK_LAYER_LUNARG_core_validation"}};

	for (auto &validation_layers : validation_layer_priority_list)
	{
		if (validate_layers(validation_layers, supported_instance_layers))
		{
			return validation_layers;
		}

		LOGW("Couldn't enable validation layers (see log for error) - falling back");
	}

	// Else return nothing
	return {};
}

/**
 * @brief Initializes the Vulkan instance.
 *
 * @param context A newly created Vulkan context.
 * @param required_instance_extensions The required Vulkan instance extensions.
 * @param required_validation_layers The required Vulkan validation layers
 */
void flutter::init_instance(Context                         &ctx,
                            const std::vector<const char *> &required_instance_extensions,
                            const std::vector<const char *> &required_validation_layers) const
{
	LOGI("Initializing vulkan instance.");

	static vk::DynamicLoader dl;
	auto                     vkGetInstanceProcAddr =
	    dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
	VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

	std::vector<vk::ExtensionProperties> instance_extensions = vk::enumerateInstanceExtensionProperties();

	std::vector<const char *> active_instance_extensions(required_instance_extensions);

#if defined(VKB_DEBUG) || defined(VKB_VALIDATION_LAYERS)
	active_instance_extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
#endif

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
	active_instance_extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WIN32_KHR)
	active_instance_extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_METAL_EXT)
	active_instance_extensions.push_back(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XCB_KHR)
	active_instance_extensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XLIB_KHR)
	active_instance_extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
	active_instance_extensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_DISPLAY_KHR)
	active_instance_extensions.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
#else
#	pragma error Platform not supported
#endif

	if (!validate_extensions(active_instance_extensions, instance_extensions))
	{
		throw std::runtime_error("Required instance extensions are missing.");
	}

	std::vector<vk::LayerProperties> supported_validation_layers = vk::enumerateInstanceLayerProperties();

	std::vector<const char *> requested_validation_layers(required_validation_layers);

#ifdef VKB_VALIDATION_LAYERS
	// Determine the optimal validation layers to enable that are necessary for useful debugging
	std::vector<const char *> optimal_validation_layers = get_optimal_validation_layers(supported_validation_layers);
	requested_validation_layers.insert(requested_validation_layers.end(), optimal_validation_layers.begin(), optimal_validation_layers.end());
#endif

	if (validate_layers(requested_validation_layers, supported_validation_layers))
	{
		LOGI("Enabled Validation Layers:")
		for (const auto &layer : requested_validation_layers)
		{
			LOGI("	\t{}", layer);
		}
	}
	else
	{
		throw std::runtime_error("Required validation layers are missing.");
	}

	vk::ApplicationInfo app("Vulkan Flutter Embedder", {}, "Vulkan Samples", VK_MAKE_VERSION(1, 0, 0));

	vk::InstanceCreateInfo instance_info({}, &app, requested_validation_layers, active_instance_extensions);

#if defined(VKB_DEBUG) || defined(VKB_VALIDATION_LAYERS)
	vk::DebugReportCallbackCreateInfoEXT debug_report_create_info(vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning, debug_callback);

	instance_info.pNext = &debug_report_create_info;
#endif

	// Create the Vulkan instance
	ctx.instance = vk::createInstance(instance_info);

	// initialize function pointers for instance
	VULKAN_HPP_DEFAULT_DISPATCHER.init(context.instance);

#if defined(VKB_DEBUG) || defined(VKB_VALIDATION_LAYERS)
	ctx.debug_callback = context.instance.createDebugReportCallbackEXT(debug_report_create_info);
#endif
}

/**
 * @brief Select a physical device.
 *
 * @param context A Vulkan context with an instance already set up.
 * @param platform The platform the application is being run on
 */
void flutter::select_physical_device_and_surface(Context &context, vkb::Platform &platform)
{
	std::vector<vk::PhysicalDevice> gpus = context.instance.enumeratePhysicalDevices();

	for (size_t i = 0; i < gpus.size() && (context.graphics_queue_index < 0); i++)
	{
		context.gpu = gpus[i];

		std::vector<vk::QueueFamilyProperties> queue_family_properties = context.gpu.getQueueFamilyProperties();

		if (queue_family_properties.empty())
		{
			throw std::runtime_error("No queue family found.");
		}

		if (context.surface)
		{
			context.instance.destroySurfaceKHR(context.surface);
		}
		context.surface = platform.get_window().create_surface(context.instance, context.gpu);

		for (uint32_t j = 0; j < vkb::to_u32(queue_family_properties.size()); j++)
		{
			vk::Bool32 supports_present = context.gpu.getSurfaceSupportKHR(j, context.surface);

			// Find a queue family which supports graphics and presentation.
			if ((queue_family_properties[j].queueFlags & vk::QueueFlagBits::eGraphics) && supports_present)
			{
				context.graphics_queue_index = j;
				break;
			}
		}
	}

	if (context.graphics_queue_index < 0)
	{
		LOGE("Did not find suitable queue which supports graphics and presentation.");
	}
}

/**
 * @brief Initializes the logical device.
 *
 * @param context A Vulkan context with an instance already set up.
 * @param required_device_extensions The required Vulkan device extensions.
 */
void flutter::init_device(Context                         &ctx,
                          const std::vector<const char *> &required_device_extensions)
{
	LOGI("Initializing vulkan device.");

	std::vector<vk::ExtensionProperties> device_extensions = ctx.gpu.enumerateDeviceExtensionProperties();

	if (!validate_extensions(required_device_extensions, device_extensions))
	{
		throw std::runtime_error("Required device extensions are missing, will try without.");
	}

	float queue_priority = 1.0f;

	// Create one queue
	vk::DeviceQueueCreateInfo queue_info({}, ctx.graphics_queue_index, 1, &queue_priority);

	vk::DeviceCreateInfo device_info({}, queue_info, {}, required_device_extensions);

	ctx.device = ctx.gpu.createDevice(device_info);
	// initialize function pointers for device
	VULKAN_HPP_DEFAULT_DISPATCHER.init(ctx.device);

	ctx.queue = ctx.device.getQueue(ctx.graphics_queue_index, 0);
}

/**
 * @brief Initializes per frame data.
 * @param context A newly created Vulkan context.
 * @param per_frame The data of a frame.
 */
void flutter::init_per_frame(Context &ctx, PerFrame &per_frame)
{
	per_frame.queue_submit_fence = ctx.device.createFence({vk::FenceCreateFlagBits::eSignaled});

	vk::CommandPoolCreateInfo cmd_pool_info(vk::CommandPoolCreateFlagBits::eTransient, ctx.graphics_queue_index);
	per_frame.primary_command_pool = ctx.device.createCommandPool(cmd_pool_info);

	vk::CommandBufferAllocateInfo cmd_buf_info(per_frame.primary_command_pool, vk::CommandBufferLevel::ePrimary, 1);
	per_frame.primary_command_buffer = ctx.device.allocateCommandBuffers(cmd_buf_info).front();

	per_frame.swapchain_acquire_semaphore = ctx.device.createSemaphore({});
	per_frame.swapchain_release_semaphore = ctx.device.createSemaphore({});

	per_frame.device      = ctx.device;
	per_frame.queue_index = ctx.graphics_queue_index;
}

/**
 * @brief Tears down the frame data.
 * @param context The Vulkan context.
 * @param per_frame The data of a frame.
 */
void flutter::teardown_per_frame(Context &ctx, PerFrame &per_frame)
{
	if (per_frame.queue_submit_fence)
	{
		ctx.device.destroyFence(per_frame.queue_submit_fence);

		per_frame.queue_submit_fence = nullptr;
	}

	if (per_frame.primary_command_buffer)
	{
		ctx.device.freeCommandBuffers(per_frame.primary_command_pool, per_frame.primary_command_buffer);

		per_frame.primary_command_buffer = nullptr;
	}

	if (per_frame.primary_command_pool)
	{
		ctx.device.destroyCommandPool(per_frame.primary_command_pool);

		per_frame.primary_command_pool = nullptr;
	}

	if (per_frame.swapchain_acquire_semaphore)
	{
		ctx.device.destroySemaphore(per_frame.swapchain_acquire_semaphore);

		per_frame.swapchain_acquire_semaphore = nullptr;
	}

	if (per_frame.swapchain_release_semaphore)
	{
		ctx.device.destroySemaphore(per_frame.swapchain_release_semaphore);

		per_frame.swapchain_release_semaphore = nullptr;
	}

	per_frame.device      = nullptr;
	per_frame.queue_index = -1;
}

/**
 * @brief Initializes the Vulkan swapchain.
 * @param context A Vulkan context with a physical device already set up.
 */
void flutter::init_swapchain(Context &ctx)
{
	vk::SurfaceCapabilitiesKHR surface_properties = ctx.gpu.getSurfaceCapabilitiesKHR(ctx.surface);

	std::vector<vk::SurfaceFormatKHR> formats = ctx.gpu.getSurfaceFormatsKHR(ctx.surface);

	vk::SurfaceFormatKHR format;
	if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined)
	{
		// There is no preferred format, so pick a default one
		format        = formats[0];
		format.format = vk::Format::eR8G8B8A8Unorm;
	}
	else
	{
		if (formats.empty())
		{
			throw std::runtime_error("Surface has no formats.");
		}

		format.format = vk::Format::eUndefined;
		for (auto &candidate : formats)
		{
			switch (candidate.format)
			{
				case vk::Format::eR8G8B8A8Unorm:
				case vk::Format::eB8G8R8A8Unorm:
				case vk::Format::eA8B8G8R8UnormPack32:
					format = candidate;
					break;

				default:
					break;
			}

			if (format.format != vk::Format::eUndefined)
			{
				break;
			}
		}

		if (format.format == vk::Format::eUndefined)
		{
			format = formats[0];
		}
	}

	vk::Extent2D swapchain_size;
	if (surface_properties.currentExtent.width == 0xFFFFFFFF)
	{
		swapchain_size.width  = ctx.swapchain_dimensions.width;
		swapchain_size.height = ctx.swapchain_dimensions.height;
	}
	else
	{
		swapchain_size = surface_properties.currentExtent;
	}

	// FIFO must be supported by all implementations.
	vk::PresentModeKHR swapchain_present_mode = vk::PresentModeKHR::eFifo;

	// Determine the number of vk::Image's to use in the swapchain.
	// Ideally, we desire to own 1 image at a time, the rest of the images can
	// either be rendered to and/or being queued up for display.
	uint32_t desired_swapchain_images = surface_properties.minImageCount + 1;
	if ((surface_properties.maxImageCount > 0) && (desired_swapchain_images > surface_properties.maxImageCount))
	{
		// Application must settle for fewer images than desired.
		desired_swapchain_images = surface_properties.maxImageCount;
	}

	// Figure out a suitable surface transform.
	vk::SurfaceTransformFlagBitsKHR pre_transform =
	    (surface_properties.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity) ? vk::SurfaceTransformFlagBitsKHR::eIdentity : surface_properties.currentTransform;

	vk::SwapchainKHR old_swapchain = ctx.swapchain;

	// Find a supported composite type.
	vk::CompositeAlphaFlagBitsKHR composite = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	if (surface_properties.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque)
	{
		composite = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	}
	else if (surface_properties.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eInherit)
	{
		composite = vk::CompositeAlphaFlagBitsKHR::eInherit;
	}
	else if (surface_properties.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePreMultiplied)
	{
		composite = vk::CompositeAlphaFlagBitsKHR::ePreMultiplied;
	}
	else if (surface_properties.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePostMultiplied)
	{
		composite = vk::CompositeAlphaFlagBitsKHR::ePostMultiplied;
	}

	vk::SwapchainCreateInfoKHR info;
	info.surface            = ctx.surface;
	info.minImageCount      = desired_swapchain_images;
	info.imageFormat        = format.format;
	info.imageColorSpace    = format.colorSpace;
	info.imageExtent.width  = swapchain_size.width;
	info.imageExtent.height = swapchain_size.height;
	info.imageArrayLayers   = 1;
	info.imageUsage         = vk::ImageUsageFlagBits::eColorAttachment;
	info.imageSharingMode   = vk::SharingMode::eExclusive;
	info.preTransform       = pre_transform;
	info.compositeAlpha     = composite;
	info.presentMode        = swapchain_present_mode;
	info.clipped            = true;
	info.oldSwapchain       = old_swapchain;

	ctx.swapchain = ctx.device.createSwapchainKHR(info);

	if (old_swapchain)
	{
		for (vk::ImageView image_view : ctx.swapchain_image_views)
		{
			ctx.device.destroyImageView(image_view);
		}

		size_t image_count = ctx.device.getSwapchainImagesKHR(old_swapchain).size();

		for (size_t i = 0; i < image_count; i++)
		{
			teardown_per_frame(ctx, ctx.per_frame[i]);
		}

		ctx.swapchain_image_views.clear();

		ctx.device.destroySwapchainKHR(old_swapchain);
	}

	ctx.swapchain_dimensions = {swapchain_size.width, swapchain_size.height, format.format};

	/// The swapchain images.
	context.swapchain_images_ = ctx.device.getSwapchainImagesKHR(ctx.swapchain);
	size_t                 image_count      = context.swapchain_images_.size();

	// Initialize per-frame resources.
	// Every swapchain image has its own command pool and fence manager.
	// This makes it very easy to keep track of when we can reset command buffers and such.
	ctx.per_frame.clear();
	ctx.per_frame.resize(image_count);

	for (size_t i = 0; i < image_count; i++)
	{
		init_per_frame(ctx, ctx.per_frame[i]);
	}

	vk::ImageViewCreateInfo view_info;
	view_info.viewType                    = vk::ImageViewType::e2D;
	view_info.format                      = ctx.swapchain_dimensions.format;
	view_info.subresourceRange.levelCount = 1;
	view_info.subresourceRange.layerCount = 1;
	view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	view_info.components.r                = vk::ComponentSwizzle::eR;
	view_info.components.g                = vk::ComponentSwizzle::eG;
	view_info.components.b                = vk::ComponentSwizzle::eB;
	view_info.components.a                = vk::ComponentSwizzle::eA;
	for (size_t i = 0; i < image_count; i++)
	{
		// Create an image view which we can render into.
		view_info.image = context.swapchain_images_[i];

		ctx.swapchain_image_views.push_back(ctx.device.createImageView(view_info));
	}
}

/**
 * @brief Initializes the Vulkan render pass.
 * @param context A Vulkan context with a device already set up.
 */
void flutter::init_render_pass(Context &ctx)
{
	vk::AttachmentDescription attachment;
	// Backbuffer format.
	attachment.format = ctx.swapchain_dimensions.format;
	// Not multisampled.
	attachment.samples = vk::SampleCountFlagBits::e1;
	// When starting the frame, we want tiles to be cleared.
	attachment.loadOp = vk::AttachmentLoadOp::eClear;
	// When ending the frame, we want tiles to be written out.
	attachment.storeOp = vk::AttachmentStoreOp::eStore;
	// Don't care about stencil since we're not using it.
	attachment.stencilLoadOp  = vk::AttachmentLoadOp::eDontCare;
	attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;

	// The image layout will be undefined when the render pass begins.
	attachment.initialLayout = vk::ImageLayout::eUndefined;
	// After the render pass is complete, we will transition to ePresentSrcKHR layout.
	attachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

	// We have one subpass. This subpass has one color attachment.
	// While executing this subpass, the attachment will be in attachment optimal layout.
	vk::AttachmentReference color_ref(0, vk::ImageLayout::eColorAttachmentOptimal);

	// We will end up with two transitions.
	// The first one happens right before we start subpass #0, where
	// eUndefined is transitioned into eColorAttachmentOptimal.
	// The final layout in the render pass attachment states ePresentSrcKHR, so we
	// will get a final transition from eColorAttachmentOptimal to ePresetSrcKHR.
	vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, color_ref);

	// Create a dependency to external events.
	// We need to wait for the WSI semaphore to signal.
	// Only pipeline stages which depend on eColorAttachmentOutput will
	// actually wait for the semaphore, so we must also wait for that pipeline stage.
	vk::SubpassDependency dependency;
	dependency.srcSubpass   = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass   = 0;
	dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;

	// Since we changed the image layout, we need to make the memory visible to
	// color attachment to modify.
	dependency.srcAccessMask = {};
	dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;

	// Finally, create the renderpass.
	vk::RenderPassCreateInfo rp_info({}, attachment, subpass, dependency);

	ctx.render_pass = ctx.device.createRenderPass(rp_info);
}

/**
 * @brief Helper function to load a shader module.
 * @param context A Vulkan context with a device.
 * @param path The path for the shader (relative to the assets directory).
 * @returns A vk::ShaderModule handle. Aborts execution if shader creation fails.
 */
vk::ShaderModule flutter::load_shader_module(Context &ctx, const char *path)
{
	vkb::GLSLCompiler glsl_compiler;

	auto buffer = vkb::fs::read_shader_binary(path);

	std::string file_ext = path;

	// Extract extension name from the glsl shader file
	file_ext = file_ext.substr(file_ext.find_last_of('.') + 1);

	std::vector<uint32_t> spirv;
	std::string           info_log;

	// Compile the GLSL source
	if (!glsl_compiler.compile_to_spirv(find_shader_stage(file_ext), buffer, "main", {}, spirv, info_log))
	{
		LOGE("Failed to compile shader, Error: {}", info_log.c_str());
		return nullptr;
	}

	vk::ShaderModuleCreateInfo module_info({}, spirv);

	return ctx.device.createShaderModule(module_info);
}

/**
 * @brief Initializes the Vulkan pipeline.
 * @param context A Vulkan context with a device and a render pass already set up.
 */
void flutter::init_pipeline(Context &ctx)
{
	// Create a blank pipeline layout.
	// We are not binding any resources to the pipeline in this first sample.
	ctx.pipeline_layout = ctx.device.createPipelineLayout({});

	vk::PipelineVertexInputStateCreateInfo vertex_input;

	// Specify we will use triangle lists to draw geometry.
	vk::PipelineInputAssemblyStateCreateInfo input_assembly({}, vk::PrimitiveTopology::eTriangleList);

	// Specify rasterization state.
	vk::PipelineRasterizationStateCreateInfo raster;
	raster.cullMode  = vk::CullModeFlagBits::eBack;
	raster.frontFace = vk::FrontFace::eClockwise;
	raster.lineWidth = 1.0f;

	// Our attachment will write to all color channels, but no blending is enabled.
	vk::PipelineColorBlendAttachmentState blend_attachment;
	blend_attachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

	vk::PipelineColorBlendStateCreateInfo blend({}, {}, {}, blend_attachment);

	// We will have one viewport and scissor box.
	vk::PipelineViewportStateCreateInfo viewport;
	viewport.viewportCount = 1;
	viewport.scissorCount  = 1;

	// Disable all depth testing.
	vk::PipelineDepthStencilStateCreateInfo depth_stencil;

	// No multisampling.
	vk::PipelineMultisampleStateCreateInfo multisample({}, vk::SampleCountFlagBits::e1);

	// Specify that these states will be dynamic, i.e. not part of pipeline state object.
	std::array<vk::DynamicState, 2> dynamics{vk::DynamicState::eViewport, vk::DynamicState::eScissor};

	vk::PipelineDynamicStateCreateInfo dynamic({}, dynamics);

	// Load our SPIR-V shaders.
	std::array<vk::PipelineShaderStageCreateInfo, 2> shader_stages{
	    vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, load_shader_module(ctx, "triangle.vert"), "main"),
	    vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, load_shader_module(ctx, "triangle.frag"), "main")};

	vk::GraphicsPipelineCreateInfo pipe({}, shader_stages);
	pipe.pVertexInputState   = &vertex_input;
	pipe.pInputAssemblyState = &input_assembly;
	pipe.pRasterizationState = &raster;
	pipe.pColorBlendState    = &blend;
	pipe.pMultisampleState   = &multisample;
	pipe.pViewportState      = &viewport;
	pipe.pDepthStencilState  = &depth_stencil;
	pipe.pDynamicState       = &dynamic;

	// We need to specify the pipeline layout and the render pass description up front as well.
	pipe.renderPass = ctx.render_pass;
	pipe.layout     = ctx.pipeline_layout;

	ctx.pipeline = ctx.device.createGraphicsPipeline(nullptr, pipe).value;

	// Pipeline is baked, we can delete the shader modules now.
	ctx.device.destroyShaderModule(shader_stages[0].module);
	ctx.device.destroyShaderModule(shader_stages[1].module);
}

/**
 * @brief Acquires an image from the swapchain.
 * @param context A Vulkan context with a swapchain already set up.
 * @param[out] image The swapchain index for the acquired image.
 * @returns Vulkan result code
 */
vk::Result flutter::acquire_next_image(Context &ctx, uint32_t *image)
{
	vk::Semaphore acquire_semaphore;
	if (ctx.recycled_semaphores.empty())
	{
		acquire_semaphore = ctx.device.createSemaphore({});
	}
	else
	{
		acquire_semaphore = ctx.recycled_semaphores.back();
		ctx.recycled_semaphores.pop_back();
	}

	vk::Result res;
	std::tie(res, *image) = ctx.device.acquireNextImageKHR(ctx.swapchain, UINT64_MAX, acquire_semaphore);

	if (res != vk::Result::eSuccess)
	{
		ctx.recycled_semaphores.push_back(acquire_semaphore);
		return res;
	}

	// If we have outstanding fences for this swapchain image, wait for them to complete first.
	// After begin frame returns, it is safe to reuse or delete resources which
	// were used previously.
	//
	// We wait for fences which completes N frames earlier, so we do not stall,
	// waiting for all GPU work to complete before this returns.
	// Normally, this doesn't really block at all,
	// since we're waiting for old frames to have been completed, but just in case.
	if (ctx.per_frame[*image].queue_submit_fence)
	{
		ctx.device.waitForFences(ctx.per_frame[*image].queue_submit_fence, true, UINT64_MAX);
		ctx.device.resetFences(ctx.per_frame[*image].queue_submit_fence);
	}

	if (ctx.per_frame[*image].primary_command_pool)
	{
		ctx.device.resetCommandPool(ctx.per_frame[*image].primary_command_pool);
	}

	// Recycle the old semaphore back into the semaphore manager.
	vk::Semaphore old_semaphore = ctx.per_frame[*image].swapchain_acquire_semaphore;

	if (old_semaphore)
	{
		ctx.recycled_semaphores.push_back(old_semaphore);
	}

	ctx.per_frame[*image].swapchain_acquire_semaphore = acquire_semaphore;

	return vk::Result::eSuccess;
}

/**
 * @brief Presents an image to the swapchain.
 * @param context The Vulkan context, with a swapchain and per-frame resources already set up.
 * @param index The swapchain index previously obtained from @ref acquire_next_image.
 * @returns Vulkan result code
 */
vk::Result flutter::present_image(Context &ctx, uint32_t index)
{
	vk::PresentInfoKHR present(ctx.per_frame[index].swapchain_release_semaphore, ctx.swapchain, index);
	return ctx.queue.presentKHR(present);
}

/**
 * @brief Initializes the Vulkan framebuffers.
 * @param context A Vulkan context with the render pass already set up.
 */
void flutter::init_framebuffers(Context &ctx)
{
	vk::Device device = ctx.device;

	// Create framebuffer for each swapchain image view
	for (auto &image_view : ctx.swapchain_image_views)
	{
		// Build the framebuffer.
		vk::FramebufferCreateInfo fb_info({}, ctx.render_pass, image_view, ctx.swapchain_dimensions.width, ctx.swapchain_dimensions.height, 1);

		ctx.swapchain_framebuffers.push_back(device.createFramebuffer(fb_info));
	}
}

/**
 * @brief Tears down the framebuffers. If our swapchain changes, we will call this, and create a new swapchain.
 * @param context The Vulkan context.
 */
void flutter::teardown_framebuffers(Context &ctx)
{
	// Wait until device is idle before teardown.
	ctx.queue.waitIdle();

	for (auto &framebuffer : ctx.swapchain_framebuffers)
	{
		ctx.device.destroyFramebuffer(framebuffer);
	}

	ctx.swapchain_framebuffers.clear();
}

/**
 * @brief Tears down the Vulkan context.
 * @param context The Vulkan context.
 */
void flutter::teardown(Context &ctx)
{
	// Don't release anything until the GPU is completely idle.
	ctx.device.waitIdle();

	teardown_framebuffers(ctx);

	for (auto &per_frame : ctx.per_frame)
	{
		teardown_per_frame(ctx, per_frame);
	}

	ctx.per_frame.clear();

	for (auto semaphore : ctx.recycled_semaphores)
	{
		ctx.device.destroySemaphore(semaphore);
	}

	if (ctx.pipeline)
	{
		ctx.device.destroyPipeline(ctx.pipeline);
	}

	if (ctx.pipeline_layout)
	{
		ctx.device.destroyPipelineLayout(ctx.pipeline_layout);
	}

	if (ctx.render_pass)
	{
		ctx.device.destroyRenderPass(ctx.render_pass);
	}

	for (auto image_view : ctx.swapchain_image_views)
	{
		ctx.device.destroyImageView(image_view);
	}

	if (ctx.swapchain)
	{
		ctx.device.destroySwapchainKHR(ctx.swapchain);
		ctx.swapchain = nullptr;
	}

	if (ctx.surface)
	{
		ctx.instance.destroySurfaceKHR(ctx.surface);
		ctx.surface = nullptr;
	}

	if (ctx.device)
	{
		ctx.device.destroy();
		ctx.device = nullptr;
	}

	if (ctx.debug_callback)
	{
		ctx.instance.destroyDebugReportCallbackEXT(ctx.debug_callback);
		ctx.debug_callback = nullptr;
	}

	ctx.instance.destroy();
}

void flutter::PlatformMessageCallback(
    const FlutterPlatformMessage *message, void *user_data)
{
	auto obj = static_cast<flutter *>(user_data);

	auto callback = obj->context.platform_message_handlers[message->channel];
    if(callback) {
		callback(message, user_data);
	} else {
		std::string msg(reinterpret_cast<const char*>(message->message));
		msg.resize(message->message_size);
		LOGI("PlatformMessage: [{}] ({}) \"{}\"", message->channel, message->message_size, msg);
		obj->context.engine_proc_table_.SendPlatformMessageResponse(obj->context.engine_, message->response_handle, nullptr, 0);
	}
}

flutter::flutter()
{
	context.pending_tasks_ = std::make_unique<
	std::priority_queue<
	    std::pair<uint64_t, FlutterTask>,
	    std::vector<std::pair<uint64_t, FlutterTask>>,
	    Context::CompareFlutterTask>>();

	context.command_line_args_c_.emplace_back("flutter");
	context.command_line_args_c_.emplace_back("--verbose-logging");
	//context.command_line_args_c_.emplace_back("--disable-observatory");
}

flutter::~flutter()
{
	if (context.running_)
	{
		context.engine_proc_table_.Shutdown(context.engine_);
		context.running_ = false;
	}
	context.engine_proc_table_.Deinitialize(context.engine_);
	teardown(context);
}

flutter::UniqueAotDataPtr flutter::LoadAotData(const std::string &aot_data_path) const
{
	LOGI("Loading AOT: {}", aot_data_path);

	FlutterEngineAOTDataSource source{};
	source.type     = kFlutterEngineAOTDataSourceTypeElfPath;
	source.elf_path = aot_data_path.c_str();

	FlutterEngineAOTData data;

	if (kSuccess != context.engine_proc_table_.CreateAOTData(&source, &data))
	{
		LOGE("Failed to load AOT data from: {}", aot_data_path);
		return nullptr;
	}
	return UniqueAotDataPtr(data);
}

FlutterEngineResult flutter::RunTask()
{
	if (!context.engine_)
	{
		return kSuccess;
	}

	if (!context.pending_tasks_->empty())
	{
		uint64_t current = context.engine_proc_table_.GetCurrentTime();
		if (current >= context.pending_tasks_->top().first)
		{
			auto item = context.pending_tasks_->top();
			context.pending_tasks_->pop();
			return context.engine_proc_table_.RunTask(context.engine_, &item.second);
		}
	}
	return kSuccess;
}

void flutter::configure_locales() const
{
#if defined(__linux__)
	// for now only support user locale
	std::string user_locale = std::locale("").name();

	// TODO - check for '*'

	std::string language = user_locale.substr(0, 2);
	std::string country  = user_locale.substr(3, 2);

	LOGI("language: {}", language);
	LOGI("country: {}", country);

	// Set available system locales
	FlutterLocale locale = {
	    .struct_size   = sizeof(locale),
	    .language_code = language.c_str(),
	    .country_code  = country.c_str(),
	    .script_code   = "",
	    .variant_code  = nullptr};

	std::vector<const FlutterLocale *> locales;
	locales.push_back(&locale);

	if (kSuccess != context.engine_proc_table_.UpdateLocales(
	                    context.engine_,
	                    locales.data(),
	                    locales.size()))
	{
		LOGE("Failed to set Flutter Engine Locale");
	}
#endif
}

bool flutter::prepare(vkb::Platform &platform)
{
	init_instance(context, {VK_KHR_SURFACE_EXTENSION_NAME}, {});
	select_physical_device_and_surface(context, platform);

	const auto &extent                  = platform.get_window().get_extent();
	context.swapchain_dimensions.width  = extent.width;
	context.swapchain_dimensions.height = extent.height;

	init_device(context, {VK_KHR_SWAPCHAIN_EXTENSION_NAME});

	init_swapchain(context);

	// Create the necessary objects for rendering.
//	init_render_pass(context);
//	init_pipeline(context);
//	init_framebuffers(context);

	context.event_loop_thread_ = pthread_self();

	context.engine_proc_table_.struct_size = sizeof(FlutterEngineProcTable);
	if (kSuccess != FlutterEngineGetProcAddresses(&context.engine_proc_table_))
	{
		LOGE("FlutterEngineGetProcAddresses != kSuccess");
	}

	context.project_args_.struct_size   = sizeof(FlutterProjectArgs);
	context.project_args_.assets_path   = "/home/joel/development/gallery/build/flutter_assets";
	context.project_args_.icu_data_path = "/usr/local/share/flutter/icudtl.dat";
	if (context.engine_proc_table_.RunsAOTCompiledDartCode())
	{
		LOGI("Runtime != Debug");
		aot_data_ = LoadAotData("/usr/local/share/homescreen/bundle/flutter_assets/libapp.so");
		if (aot_data_ && aot_data_.get())
		{
			context.project_args_.aot_data = aot_data_.get();
		}
	}
	else
	{
		LOGI("Runtime == Debug");
	}

	context.project_args_.persistent_cache_path         = "/home/joel/.vkflutter";
	context.project_args_.is_persistent_cache_read_only = false;
	context.project_args_.shutdown_dart_vm_when_done    = true;
	context.project_args_.command_line_argc             = static_cast<int>(context.command_line_args_c_.size());
	context.project_args_.command_line_argv             = context.command_line_args_c_.data();
	context.project_args_.log_message_callback =
	    [](const char *tag, const char *message, void *user_data) {
		    LOGI("{}", message);
	    };
	context.project_args_.compute_platform_resolved_locale_callback =
	    [](const FlutterLocale **supported_locales, size_t number_of_locales) -> const FlutterLocale * {
		LOGI("locales_count: {}", number_of_locales);
		for (int i = 0; i < number_of_locales; i++)
		{
			LOGI("language_code: {}", supported_locales[i]->language_code);
			LOGI("country_code: {}", supported_locales[i]->country_code);
			LOGI("script_code: {}", supported_locales[i]->script_code);
			LOGI("variant_code: {}", supported_locales[i]->variant_code);
		}
		return supported_locales[0];
	};
	context.project_args_.platform_message_callback =
	    [](const FlutterPlatformMessage *message, void *user_data) {
		    auto obj = static_cast<flutter *>(user_data);

		    auto callback = obj->context.platform_message_handlers[message->channel];
		    if(callback) {
			    callback(message, user_data);
		    } else {
			    std::string msg(reinterpret_cast<const char*>(message->message));
			    msg.resize(message->message_size);
			    LOGI("PlatformMessage: [{}] ({}) \"{}\"", message->channel, message->message_size, msg);
			    obj->context.engine_proc_table_.SendPlatformMessageResponse(obj->context.engine_, message->response_handle, nullptr, 0);
		    }
	    };
	context.project_args_.on_pre_engine_restart_callback =
	    [](void *user_data) {
		    LOGI("on pre engine restart");
	    };
#if 0
	context.project_args_.root_isolate_create_callback =
	    [](void *user_data) {
		    LOGI("root isolate created");
	    };
#endif
#if 0
	context.project_args_.vsync_callback =
	    [](void *user_data, intptr_t baton) {
		    reinterpret_cast<flutter *>(user_data)->context.vsync_baton = baton;
	    };
#endif
	context.platform_task_runner_.struct_size                          = sizeof(FlutterTaskRunnerDescription);
	context.platform_task_runner_.user_data                            = this;
	context.platform_task_runner_.identifier                           = 1UL;
	context.platform_task_runner_.runs_task_on_current_thread_callback = [](void *user_data) -> bool {
		return pthread_equal(pthread_self(), static_cast<flutter *>(user_data)->context.event_loop_thread_) != 0;
	};
	context.platform_task_runner_.post_task_callback = [](FlutterTask task, uint64_t target_time, void *user_data) -> void {
		auto obj = static_cast<flutter *>(user_data);
        obj->PostTaskCallback(task, target_time, obj);
	};

	context.custom_task_runners_.struct_size          = sizeof(FlutterCustomTaskRunners);
	context.custom_task_runners_.platform_task_runner = &context.platform_task_runner_,
	context.project_args_.custom_task_runners         = &context.custom_task_runners_;

	context.renderer_config_.type                      = kVulkan;
	context.renderer_config_.vulkan.struct_size        = sizeof(FlutterVulkanRendererConfig);
	context.renderer_config_.vulkan.instance           = context.instance;
	context.renderer_config_.vulkan.physical_device    = context.gpu;
	context.renderer_config_.vulkan.device             = context.device;
	context.renderer_config_.vulkan.queue_family_index = context.graphics_queue_index;
	context.renderer_config_.vulkan.queue              = context.queue;

	context.renderer_config_.vulkan.get_instance_proc_address_callback =
	    [](void *user_data, FlutterVulkanInstanceHandle instance,
	       const char *name) -> void * {
		return reinterpret_cast<void *>(reinterpret_cast<flutter *>(user_data)->context.instance.getProcAddr(name));
	};

	context.renderer_config_.vulkan.get_next_image_callback =
	    [](void *user_data, const FlutterFrameInfo *frame_info) -> FlutterVulkanImage {
		auto obj = reinterpret_cast<flutter *>(user_data);

		auto     res = obj->acquire_next_image(obj->context, &obj->context.frame_info_.index);

		// Handle outdated error in acquire.
		if (res == vk::Result::eSuboptimalKHR || res == vk::Result::eErrorOutOfDateKHR)
		{
			obj->resize(obj->context.swapchain_dimensions.width, obj->context.swapchain_dimensions.height);
			res = obj->acquire_next_image(obj->context, &obj->context.frame_info_.index);
		}

		if (res != vk::Result::eSuccess)
		{
			obj->context.queue.waitIdle();
			return {
			    .struct_size = sizeof(FlutterVulkanImage),
			    .image       = nullptr};
		}

		VkImage vulkan_image = obj->context.swapchain_images_[obj->context.frame_info_.index];
		return {
		    .struct_size = sizeof(FlutterVulkanImage),
		    .image       = vulkan_image,
		};
	};

	context.renderer_config_.vulkan.present_image_callback =
	    [](void *user_data, const FlutterVulkanImage *image) -> bool {

		auto obj = reinterpret_cast<flutter *>(user_data);
		auto res = obj->present_image(obj->context, obj->context.frame_info_.index);

		// Handle Outdated error in present.
		if (res == vk::Result::eSuboptimalKHR || res == vk::Result::eErrorOutOfDateKHR)
		{
			obj->resize(obj->context.swapchain_dimensions.width, obj->context.swapchain_dimensions.height);
		}
		else if (res != vk::Result::eSuccess)
		{
			LOGE("Failed to present swapchain image.");
			return false;
		}

		return true;
	};

	context.platform_message_handlers.emplace("flutter/isolate", MessageCallback_ChannelIsolate);
	context.platform_message_handlers.emplace("flutter/restoration", MessageCallback_ChannelRestoration);
	context.platform_message_handlers.emplace("flutter/platform", MessageCallback_ChannelPlatform);
	context.platform_message_handlers.emplace("flutter/navigation", MessageCallback_ChannelNavigation);

	if (kSuccess != context.engine_proc_table_.Run(FLUTTER_ENGINE_VERSION,
	                                               &context.renderer_config_,
	                                               &context.project_args_,
	                                               this,
	                                               &context.engine_))
	{
		LOGE("Run != kSuccess");
	}

	context.running_ = true;

	configure_locales();

	vk::SurfaceCapabilitiesKHR surface_properties = context.gpu.getSurfaceCapabilitiesKHR(context.surface);

	// Update Flutter Engine
	FlutterWindowMetricsEvent fwme = {
		.struct_size = sizeof(FlutterWindowMetricsEvent),
		.width       = surface_properties.currentExtent.width,
		.height      = surface_properties.currentExtent.height,
		.pixel_ratio = 1.0};

	if (kSuccess != context.engine_proc_table_.SendWindowMetricsEvent(context.engine_, &fwme))
	{
		LOGE("Failed send initial window size to flutter");
	}

	return true;
}

void flutter::update(float delta_time)
{
#if 0
	auto t1 = context.engine_proc_table_.GetCurrentTime();
#endif
	RunTask();
#if 0
	auto t2 = context.engine_proc_table_.GetCurrentTime();

	if (context.vsync_baton)
	{
		context.engine_proc_table_.OnVsync(context.engine_,
		                                   context.vsync_baton,
		                                   t1,
		                                   t2 + (16670 - (t2 - t1)));
		context.vsync_baton = 0;
	}
#endif
}

bool flutter::resize(const uint32_t, const uint32_t)
{
	if (!context.device)
	{
		return false;
	}

	vk::SurfaceCapabilitiesKHR surface_properties = context.gpu.getSurfaceCapabilitiesKHR(context.surface);

	if (context.running_)
	{
		// Update Flutter Engine
		FlutterWindowMetricsEvent fwme = {
		    .struct_size = sizeof(FlutterWindowMetricsEvent),
		    .width       = surface_properties.currentExtent.width,
		    .height      = surface_properties.currentExtent.height,
		    // TODO - use 1.0 for now
		    .pixel_ratio = 1.0};

		if (kSuccess != context.engine_proc_table_.SendWindowMetricsEvent(context.engine_, &fwme))
		{
			LOGE("Failed send initial window size to flutter");
		}
	}

	// Only rebuild the swapchain if the dimensions have changed
	if (surface_properties.currentExtent.width == context.swapchain_dimensions.width &&
	    surface_properties.currentExtent.height == context.swapchain_dimensions.height)
	{
		return false;
	}

	context.device.waitIdle();
	teardown_framebuffers(context);

	init_swapchain(context);
	init_framebuffers(context);

	return true;
}

void flutter::PostTaskCallback(FlutterTask task, uint64_t target_time, flutter *obj)
{
	obj->context.pending_tasks_->push(std::make_pair(target_time, task));
	if (!obj->context.running_) {
		uint64_t current = FlutterEngineGetCurrentTime();
		if (current >= obj->context.pending_tasks_->top().first) {
			auto item = obj->context.pending_tasks_->top();
			if (kSuccess ==
			    obj->context.engine_proc_table_.RunTask(obj->context.engine_, &item.second)) {
				obj->context.pending_tasks_->pop();
			}
		}
	}
}

}        // namespace flutter

std::unique_ptr<vkb::Application> create_flutter()
{
	return std::make_unique<flutter::flutter>();
}
