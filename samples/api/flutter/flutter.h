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

#pragma once

#include "common/vk_common.h"
#include "flutter_embedder.h"
#include "platform/application.h"
#include <vulkan/vulkan.hpp>

namespace flutter
{

/**
 * @brief A self-contained (minimal use of framework) sample that illustrates
 * the rendering of a triangle
 */
class flutter : public vkb::Application
{
	/**
	 * @brief Swapchain state
	 */
	struct SwapchainDimensions
	{
		/// Width of the swapchain.
		uint32_t width = 0;

		/// Height of the swapchain.
		uint32_t height = 0;

		/// Pixel format of the swapchain.
		vk::Format format = vk::Format::eUndefined;
	};

	/**
	 * @brief Per-frame data
	 */
	struct PerFrame
	{
		vk::Device device;

		vk::Fence queue_submit_fence;

		vk::CommandPool primary_command_pool;

		vk::CommandBuffer primary_command_buffer;

		vk::Semaphore swapchain_acquire_semaphore;

		vk::Semaphore swapchain_release_semaphore;

		int32_t queue_index = 0;
	};

	/**
	 * @brief Vulkan objects and global state
	 */
	struct Context
	{
		/// The Vulkan instance.
		vk::Instance instance;

		/// The Vulkan physical device.
		vk::PhysicalDevice gpu;

		/// The Vulkan device.
		vk::Device device;

		/// The Vulkan device queue.
		vk::Queue queue;

		/// The swapchain.
		vk::SwapchainKHR swapchain;

		/// The swapchain dimensions.
		SwapchainDimensions swapchain_dimensions;

		/// The surface we will render to.
		vk::SurfaceKHR surface;

		/// The queue family index where graphics work will be submitted.
		int32_t graphics_queue_index = -1;

		/// The image view for each swapchain image.
		std::vector<vk::ImageView> swapchain_image_views;

		/// The framebuffer for each swapchain image view.
		std::vector<vk::Framebuffer> swapchain_framebuffers;

		/// The renderpass description.
		vk::RenderPass render_pass;

		/// The graphics pipeline.
		vk::Pipeline pipeline;

		/**
		 * The pipeline layout for resources.
		 * Not used in this sample, but we still need to provide a dummy one.
		 */
		vk::PipelineLayout pipeline_layout;

		/// The debug report callback.
		vk::DebugReportCallbackEXT debug_callback;

		/// A set of semaphores that can be reused.
		std::vector<vk::Semaphore> recycled_semaphores;

		/// A set of per-frame data.
		std::vector<PerFrame> per_frame;

		/// Flutter Engine process table
		FlutterEngineProcTable engine_proc_table_{};

		/// Flutter Engine instance
		FlutterEngine engine_;

		/// Project Arguments used to initialize the flutter engine
		FlutterProjectArgs project_args_{};

		/// Renderer Configuration used to initialize the flutter engine
		FlutterRendererConfig renderer_config_{};

		/// Task Runner configuration
		FlutterTaskRunnerDescription platform_task_runner_{};
		FlutterCustomTaskRunners     custom_task_runners_{};

		/// Command line arguments passed to Engine
		std::vector<const char *> command_line_args_c_;

		std::vector<vk::Image> swapchain_images_;

		struct FrameInfo
		{
			uint32_t index;
		} frame_info_{};

		class CompareFlutterTask
		{
		  public:
			bool operator()(std::pair<uint64_t, FlutterTask> n1,
			                std::pair<uint64_t, FlutterTask> n2)
			{
				return n1.first > n2.first;
			}
		};

		pthread_mutex_t mutex_;
		std::unique_ptr<std::priority_queue<
		    std::pair<uint64_t, FlutterTask>,
		    std::vector<std::pair<uint64_t, FlutterTask>>,
		    CompareFlutterTask>> pending_tasks_;

		bool running_;

		intptr_t vsync_baton = 0;

		pthread_t event_loop_thread_;

		std::map<std::string, FlutterPlatformMessageCallback> platform_message_handlers;
	};

  public:
	flutter();

	~flutter() override;

	bool prepare(vkb::Platform &platform) override;

	void update(float delta_time) override;

	bool resize(uint32_t width, uint32_t height) override;

	static bool validate_extensions(const std::vector<const char *>            &required,
	                                const std::vector<vk::ExtensionProperties> &available);

	static VkShaderStageFlagBits find_shader_stage(const std::string &ext);

	void init_instance(Context                         &context,
	                   const std::vector<const char *> &required_instance_extensions,
	                   const std::vector<const char *> &required_validation_layers) const;

	static void select_physical_device_and_surface(Context &context, vkb::Platform &platform);

	static void init_device(Context                         &context,
	                        const std::vector<const char *> &required_device_extensions);

	static void init_per_frame(Context &context, PerFrame &per_frame);

	static void teardown_per_frame(Context &context, PerFrame &per_frame);

	void init_swapchain(Context &context);

	void init_render_pass(Context &context);

	vk::ShaderModule load_shader_module(Context &context, const char *path);

	void init_pipeline(Context &context);

	vk::Result acquire_next_image(Context &context, uint32_t *image);

	vk::Result present_image(Context &context, uint32_t index);

	static void init_framebuffers(Context &context);

	void teardown_framebuffers(Context &context);

	void teardown(Context &context);

	Context *get_context() { return &context; }

  private:
	Context context;

	void PlatformMessageCallback(const FlutterPlatformMessage *, void *);

	// Custom deleter for FlutterEngineAOTData.
	struct AOTDataDeleter
	{
		void operator()(FlutterEngineAOTData aot_data)
		{
			FlutterEngineCollectAOTData(aot_data);
		}
	};
	using UniqueAotDataPtr = std::unique_ptr<_FlutterEngineAOTData, AOTDataDeleter>;
	UniqueAotDataPtr LoadAotData(const std::string &aot_data_path) const;
	UniqueAotDataPtr aot_data_;

	FlutterEngineResult RunTask();

	static void PostTaskCallback(FlutterTask task, uint64_t target_time, flutter *obj);

	void configure_locales() const;
};

}        // namespace flutter

std::unique_ptr<vkb::Application> create_flutter();
