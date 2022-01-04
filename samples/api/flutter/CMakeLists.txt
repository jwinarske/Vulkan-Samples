# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 the "License";
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

get_filename_component(FOLDER_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_LIST_DIR} PATH)
get_filename_component(CATEGORY_NAME ${PARENT_DIR} NAME)

link_directories(/home/joel/development/engine/src/out/host_debug_unopt)

add_sample(
    ID ${FOLDER_NAME}
    FILES
        flutter/json_message_codec.cc
        flutter/json_method_codec.cc
        flutter/log_settings.cc
        flutter/log_settings_state.cc
        flutter/logging.cc
        flutter/standard_codec.cc
        flutter/shell/platform/common/text_input_model.cc
        flutter/fml/platform/linux/paths_linux.cc
        flutter/fml/platform/posix/file_posix.cc
        flutter/fml/platform/posix/paths_posix.cc
        flutter/fml/platform/posix/posix_wrappers_posix.cc
        flutter/fml/platform/posix/shared_mutex_posix.cc
#        flutter/fml/command_line.cc
        flutter/fml/file.cc
        flutter/fml/log_settings.cc
        flutter/fml/log_settings_state.cc
        flutter/fml/logging.cc
        flutter/fml/paths.cc
        flutter/fml/unique_fd.cc
    LIBS
        flutter_engine
    CATEGORY ${CATEGORY_NAME}
    AUTHOR "Arm"
    NAME "Flutter"
    DESCRIPTION "An introduction into Vulkan and its respective objects using vulkan.hpp."
    SHADER_FILES_GLSL
        "triangle.vert"
        "triangle.frag"
    )

target_include_directories(${FOLDER_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(${FOLDER_NAME} PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_TYPESAFE_CONVERSION=1)
