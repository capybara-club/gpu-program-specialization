/*
 * SPDX-FileCopyrightText: 2026 Charles Durham
 * SPDX-License-Identifier: MIT
 *
 * MIT License
 *
 * Copyright (c) 2026 Charles Durham
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#ifdef IMGUI_IMPL_VULKAN_USE_VOLK
#define VOLK_IMPLEMENTATION
#include <volk.h>
#endif

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include <kermac.hpp>
#include <kermac_cpu.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <type_traits>

#ifndef IMGUI_VULKAN_DEMO_FONT_PATH
#define IMGUI_VULKAN_DEMO_FONT_PATH "/usr/share/fonts/noto/NotoSans-Regular.ttf"
#endif

#ifndef IMGUI_VULKAN_DEMO_FALLBACK_FONT_PATH
#define IMGUI_VULKAN_DEMO_FALLBACK_FONT_PATH IMGUI_VULKAN_DEMO_FONT_PATH
#endif

static VkAllocationCallbacks* g_Allocator = nullptr;
static VkInstance g_Instance = VK_NULL_HANDLE;
static VkPhysicalDevice g_PhysicalDevice = VK_NULL_HANDLE;
static VkDevice g_Device = VK_NULL_HANDLE;
static uint32_t g_QueueFamily = (uint32_t)-1;
static VkQueue g_Queue = VK_NULL_HANDLE;
static VkPipelineCache g_PipelineCache = VK_NULL_HANDLE;
static VkDescriptorPool g_DescriptorPool = VK_NULL_HANDLE;

static ImGui_ImplVulkanH_Window g_MainWindowData;
static uint32_t g_MinImageCount = 2;
static bool g_SwapChainRebuild = false;

static void glfw_error_callback(int error, const char* description) {
    std::fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

static void check_vk_result(VkResult err) {
    if (err == VK_SUCCESS) {
        return;
    }
    std::fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
    if (err < 0) {
        std::abort();
    }
}

static bool IsExtensionAvailable(const ImVector<VkExtensionProperties>& properties, const char* extension) {
    for (const VkExtensionProperties& p : properties) {
        if (std::strcmp(p.extensionName, extension) == 0) {
            return true;
        }
    }
    return false;
}

static void SetupVulkan(ImVector<const char*> instance_extensions) {
    VkResult err;
#ifdef IMGUI_IMPL_VULKAN_USE_VOLK
    volkInitialize();
#endif

    {
        VkInstanceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

        uint32_t properties_count;
        ImVector<VkExtensionProperties> properties;
        vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, nullptr);
        properties.resize(properties_count);
        err = vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, properties.Data);
        check_vk_result(err);

        if (IsExtensionAvailable(properties, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
            instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
        if (IsExtensionAvailable(properties, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif

        create_info.enabledExtensionCount = (uint32_t)instance_extensions.Size;
        create_info.ppEnabledExtensionNames = instance_extensions.Data;
        err = vkCreateInstance(&create_info, g_Allocator, &g_Instance);
        check_vk_result(err);
#ifdef IMGUI_IMPL_VULKAN_USE_VOLK
        volkLoadInstance(g_Instance);
#endif
    }

    g_PhysicalDevice = ImGui_ImplVulkanH_SelectPhysicalDevice(g_Instance);
    IM_ASSERT(g_PhysicalDevice != VK_NULL_HANDLE);

    g_QueueFamily = ImGui_ImplVulkanH_SelectQueueFamilyIndex(g_PhysicalDevice);
    IM_ASSERT(g_QueueFamily != (uint32_t)-1);

    {
        ImVector<const char*> device_extensions;
        device_extensions.push_back("VK_KHR_swapchain");

        uint32_t properties_count;
        ImVector<VkExtensionProperties> properties;
        vkEnumerateDeviceExtensionProperties(g_PhysicalDevice, nullptr, &properties_count, nullptr);
        properties.resize(properties_count);
        vkEnumerateDeviceExtensionProperties(g_PhysicalDevice, nullptr, &properties_count, properties.Data);
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
        if (IsExtensionAvailable(properties, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
            device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        }
#endif

        const float queue_priority[] = { 1.0f };
        VkDeviceQueueCreateInfo queue_info[1] = {};
        queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info[0].queueFamilyIndex = g_QueueFamily;
        queue_info[0].queueCount = 1;
        queue_info[0].pQueuePriorities = queue_priority;

        VkDeviceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount = 1;
        create_info.pQueueCreateInfos = queue_info;
        create_info.enabledExtensionCount = (uint32_t)device_extensions.Size;
        create_info.ppEnabledExtensionNames = device_extensions.Data;
        err = vkCreateDevice(g_PhysicalDevice, &create_info, g_Allocator, &g_Device);
        check_vk_result(err);
        vkGetDeviceQueue(g_Device, g_QueueFamily, 0, &g_Queue);
    }

    {
        VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE },
        };
        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 0;
        for (VkDescriptorPoolSize& pool_size : pool_sizes) {
            pool_info.maxSets += pool_size.descriptorCount;
        }
        pool_info.poolSizeCount = (uint32_t)IM_COUNTOF(pool_sizes);
        pool_info.pPoolSizes = pool_sizes;
        err = vkCreateDescriptorPool(g_Device, &pool_info, g_Allocator, &g_DescriptorPool);
        check_vk_result(err);
    }
}

static void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface, int width, int height) {
    wd->Surface = surface;

    VkBool32 res;
    vkGetPhysicalDeviceSurfaceSupportKHR(g_PhysicalDevice, g_QueueFamily, wd->Surface, &res);
    if (res != VK_TRUE) {
        std::fprintf(stderr, "Error no WSI support on physical device 0\n");
        std::exit(-1);
    }

    const VkFormat requestSurfaceImageFormat[] = {
        VK_FORMAT_B8G8R8A8_UNORM,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_B8G8R8_UNORM,
        VK_FORMAT_R8G8B8_UNORM
    };
    const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        g_PhysicalDevice,
        wd->Surface,
        requestSurfaceImageFormat,
        (size_t)IM_COUNTOF(requestSurfaceImageFormat),
        requestSurfaceColorSpace
    );

    VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_FIFO_KHR };
    wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
        g_PhysicalDevice,
        wd->Surface,
        &present_modes[0],
        IM_COUNTOF(present_modes)
    );

    IM_ASSERT(g_MinImageCount >= 2);
    ImGui_ImplVulkanH_CreateOrResizeWindow(
        g_Instance,
        g_PhysicalDevice,
        g_Device,
        wd,
        g_QueueFamily,
        g_Allocator,
        width,
        height,
        g_MinImageCount,
        0
    );
}

static void CleanupVulkan() {
    vkDestroyDescriptorPool(g_Device, g_DescriptorPool, g_Allocator);
    vkDestroyDevice(g_Device, g_Allocator);
    vkDestroyInstance(g_Instance, g_Allocator);
}

static void CleanupVulkanWindow() {
    ImGui_ImplVulkanH_DestroyWindow(g_Instance, g_Device, &g_MainWindowData, g_Allocator);
}

static float GetWindowContentScale(GLFWwindow* window) {
    float xscale = 1.0f;
    float yscale = 1.0f;
    glfwGetWindowContentScale(window, &xscale, &yscale);
    float scale = (xscale > yscale) ? xscale : yscale;
    return (scale > 0.0f) ? scale : 1.0f;
}

static void LoadFonts(GLFWwindow* window, ImGuiIO& io) {
    const float content_scale = GetWindowContentScale(window);
    const float base_font_size = 13.0f;
    const float font_size = base_font_size * content_scale;

    io.Fonts->Clear();
    ImFont* font = io.Fonts->AddFontFromFileTTF(IMGUI_VULKAN_DEMO_FONT_PATH, font_size);
    if (font == nullptr) {
        std::fprintf(stderr, "Failed to load font: %s\n", IMGUI_VULKAN_DEMO_FONT_PATH);
        font = io.Fonts->AddFontFromFileTTF(IMGUI_VULKAN_DEMO_FALLBACK_FONT_PATH, font_size);
        if (font == nullptr) {
            std::fprintf(stderr, "Failed to load fallback font: %s\n", IMGUI_VULKAN_DEMO_FALLBACK_FONT_PATH);
            font = io.Fonts->AddFontDefault();
        }
    }
    io.FontDefault = font;
    io.Fonts->Build();
}

static void FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data) {
    VkSemaphore image_acquired_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    VkResult err = vkAcquireNextImageKHR(
        g_Device, wd->Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &wd->FrameIndex
    );
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        g_SwapChainRebuild = true;
    }
    if (err == VK_ERROR_OUT_OF_DATE_KHR) {
        return;
    }
    if (err != VK_SUBOPTIMAL_KHR) {
        check_vk_result(err);
    }

    ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
    {
        err = vkWaitForFences(g_Device, 1, &fd->Fence, VK_TRUE, UINT64_MAX);
        check_vk_result(err);
        err = vkResetFences(g_Device, 1, &fd->Fence);
        check_vk_result(err);
    }

    {
        err = vkResetCommandPool(g_Device, fd->CommandPool, 0);
        check_vk_result(err);
        VkCommandBufferBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
        check_vk_result(err);
    }

    {
        VkRenderPassBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass = wd->RenderPass;
        info.framebuffer = wd->Frames[wd->FrameIndex].Framebuffer;
        info.renderArea.extent.width = wd->Width;
        info.renderArea.extent.height = wd->Height;
        info.clearValueCount = 1;
        info.pClearValues = &wd->ClearValue;
        vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
    }

    ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

    vkCmdEndRenderPass(fd->CommandBuffer);
    {
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores = &image_acquired_semaphore;
        info.pWaitDstStageMask = &wait_stage;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &fd->CommandBuffer;
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &render_complete_semaphore;

        err = vkEndCommandBuffer(fd->CommandBuffer);
        check_vk_result(err);
        err = vkQueueSubmit(g_Queue, 1, &info, fd->Fence);
        check_vk_result(err);
    }
}

static void FramePresent(ImGui_ImplVulkanH_Window* wd) {
    if (g_SwapChainRebuild) {
        return;
    }
    VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &render_complete_semaphore;
    info.swapchainCount = 1;
    info.pSwapchains = &wd->Swapchain;
    info.pImageIndices = &wd->FrameIndex;
    VkResult err = vkQueuePresentKHR(g_Queue, &info);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        g_SwapChainRebuild = true;
        return;
    }
    check_vk_result(err);
    wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->ImageCount;
}

struct DatasetDescriptor {
    const char* name;
    const char* summary;
    int64_t max_train;
    int64_t max_test;
    int64_t image_rows;
    int64_t image_cols;
    int64_t image_channels;
    const char* const* label_names;
    int64_t num_label_names;
    const char* const* label_desc;
    const bool* label_is_int;
    bool has_split;
    bool supports_byte;
    bool byte_is_u32;
    const char* byte_label;
    bool supports_params;
    enum class LabelKind {
        ClassIndex = 0,
        FieldVector = 1
    } label_kind;
    void (*dims_fn)(
        int64_t max_num_train,
        int64_t max_num_test,
        int64_t* num_train,
        int64_t* num_test,
        int64_t* num_dims,
        int64_t* num_labels);
    void (*load_f32_fn)(
        kermac::HostTensor<float>& x_train,
        kermac::HostTensor<float>& y_train,
        kermac::HostTensor<float>& x_test,
        kermac::HostTensor<float>& y_test);
    void (*load_u8_fn)(
        kermac::HostTensor<uint8_t>& x_train,
        kermac::HostTensor<uint8_t>& y_train,
        kermac::HostTensor<uint8_t>& x_test,
        kermac::HostTensor<uint8_t>& y_test);
    void (*load_u32_fn)(
        kermac::HostTensor<uint32_t>& x_train,
        kermac::HostTensor<uint8_t>& y_train,
        kermac::HostTensor<uint32_t>& x_test,
        kermac::HostTensor<uint8_t>& y_test);
};

struct DatasetDims {
    int64_t num_train = 0;
    int64_t num_test = 0;
    int64_t num_dims = 0;
    int64_t num_labels = 0;
    int64_t image_rows = 0;
    int64_t image_cols = 0;
    int64_t image_channels = 0;
    int64_t image_plane = 0;
};

struct BufferPlan {
    bool alloc_f32_images = false;
    bool alloc_f32_labels = false;
    bool alloc_u8_images = false;
    bool alloc_u8_labels = false;
    bool alloc_u32_images = false;
};

struct DatasetBuffers {
    kermac::HostTensor<float> x_train_f32;
    kermac::HostTensor<float> y_train_f32;
    kermac::HostTensor<float> x_test_f32;
    kermac::HostTensor<float> y_test_f32;
    kermac::HostTensor<uint8_t> x_train_u8;
    kermac::HostTensor<uint8_t> y_train_u8;
    kermac::HostTensor<uint8_t> x_test_u8;
    kermac::HostTensor<uint8_t> y_test_u8;
    kermac::HostTensor<uint32_t> x_train_u32;
    kermac::HostTensor<uint32_t> x_test_u32;

    DatasetBuffers(
        kermac::HostStackAllocator& hsa,
        const DatasetDims& dims,
        const BufferPlan& plan
    )
        : x_train_f32(hsa,
                      plan.alloc_f32_images ? dims.num_train : 0,
                      plan.alloc_f32_images ? dims.num_dims : 0),
          y_train_f32(hsa,
                      plan.alloc_f32_labels ? dims.num_train : 0,
                      plan.alloc_f32_labels ? dims.num_labels : 0),
          x_test_f32(hsa,
                     plan.alloc_f32_images ? dims.num_test : 0,
                     plan.alloc_f32_images ? dims.num_dims : 0),
          y_test_f32(hsa,
                     plan.alloc_f32_labels ? dims.num_test : 0,
                     plan.alloc_f32_labels ? dims.num_labels : 0),
          x_train_u8(hsa,
                     plan.alloc_u8_images ? dims.num_train : 0,
                     plan.alloc_u8_images ? dims.image_plane : 0),
          y_train_u8(hsa,
                     plan.alloc_u8_labels ? dims.num_train : 0),
          x_test_u8(hsa,
                    plan.alloc_u8_images ? dims.num_test : 0,
                    plan.alloc_u8_images ? dims.image_plane : 0),
          y_test_u8(hsa,
                    plan.alloc_u8_labels ? dims.num_test : 0),
          x_train_u32(hsa,
                      plan.alloc_u32_images ? dims.num_train : 0,
                      plan.alloc_u32_images ? dims.image_plane : 0),
          x_test_u32(hsa,
                     plan.alloc_u32_images ? dims.num_test : 0,
                     plan.alloc_u32_images ? dims.image_plane : 0) {}
};

struct ViewerState {
    int dataset_idx = 0;
    int split_idx = 0;
    int type_idx = 0;
    int sample_idx = 0;
    float pixel_scale = 12.0f;
    int montage_rows = 4;
    int montage_cols = 4;
};

struct RelSpritesState {
    kermac::RelSpritesConfig cfg{};
};

static DatasetDims QueryDatasetDims(const DatasetDescriptor& desc) {
    DatasetDims dims = {};
    desc.dims_fn(
        desc.max_train,
        desc.max_test,
        &dims.num_train,
        &dims.num_test,
        &dims.num_dims,
        &dims.num_labels
    );
    dims.image_rows = desc.image_rows;
    dims.image_cols = desc.image_cols;
    dims.image_channels = desc.image_channels;
    dims.image_plane = desc.image_rows * desc.image_cols;
    return dims;
}

static DatasetDims RelSpritesDimsFromConfig(const kermac::RelSpritesConfig& cfg) {
    DatasetDims dims = {};
    int64_t num_samples = 0;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t channels = 0;
    int64_t num_labels = 0;
    kermac::relsprites_dims(cfg, &num_samples, &rows, &cols, &channels, &num_labels);
    dims.num_train = num_samples;
    dims.num_test = 0;
    dims.image_rows = rows;
    dims.image_cols = cols;
    dims.image_channels = channels;
    dims.image_plane = rows * cols;
    dims.num_dims = rows * cols * channels;
    dims.num_labels = num_labels;
    return dims;
}

static BufferPlan MakeBufferPlan(
    const DatasetDescriptor& dataset,
    const DatasetDims& dims,
    bool use_byte,
    bool use_u32_bytes
) {
    BufferPlan plan{};
    if (use_byte) {
        if (use_u32_bytes) {
            plan.alloc_u32_images = true;
        } else {
            plan.alloc_u8_images = true;
        }
        if (dataset.label_kind == DatasetDescriptor::LabelKind::ClassIndex) {
            plan.alloc_u8_labels = true;
        } else {
            plan.alloc_f32_labels = true;
        }
    } else {
        plan.alloc_f32_images = true;
        plan.alloc_f32_labels = true;
    }
    return plan;
}

static size_t MeasureBuffers(const DatasetDims& dims, const BufferPlan& plan) {
    kermac::HostStackAllocator hsa_dry(nullptr, 0);
    DatasetBuffers tmp(hsa_dry, dims, plan);
    return hsa_dry.get().largest_total_offset;
}

struct ActiveStorage {
    kermac::HostStackAllocator hsa;
    DatasetBuffers buffers;

    ActiveStorage(
        void* memory,
        size_t memory_bytes,
        const DatasetDims& dims,
        const BufferPlan& plan
    )
        : hsa(memory, memory_bytes),
          buffers(hsa, dims, plan) {}
};

struct ActiveStorageSlot {
    typedef typename std::aligned_storage<sizeof(ActiveStorage), alignof(ActiveStorage)>::type StorageT;
    StorageT storage;
    bool valid = false;

    ActiveStorage* get() {
        return reinterpret_cast<ActiveStorage*>(&storage);
    }

    const ActiveStorage* get() const {
        return reinterpret_cast<const ActiveStorage*>(&storage);
    }

    void reset() {
        if (valid) {
            get()->~ActiveStorage();
            valid = false;
        }
    }

    void emplace(void* memory, size_t memory_bytes, const DatasetDims& dims, const BufferPlan& plan) {
        reset();
        new (&storage) ActiveStorage(memory, memory_bytes, dims, plan);
        valid = true;
    }
};

static bool LoadDatasetF32(
    const DatasetDescriptor& desc,
    DatasetBuffers& buffers,
    char* status,
    size_t status_size
) {
    try {
        if (!desc.load_f32_fn) {
            std::snprintf(status, status_size, "No float loader for %s.", desc.name);
            return false;
        }
        desc.load_f32_fn(
            buffers.x_train_f32,
            buffers.y_train_f32,
            buffers.x_test_f32,
            buffers.y_test_f32
        );
        std::snprintf(status, status_size, "Loaded %s (float).", desc.name);
        return true;
    } catch (const std::exception& e) {
        std::snprintf(status, status_size, "Float load failed: %s", e.what());
        return false;
    }
}

static bool LoadDatasetBytes(
    const DatasetDescriptor& desc,
    DatasetBuffers& buffers,
    char* status,
    size_t status_size
) {
    try {
        if (desc.byte_is_u32) {
            if (!desc.load_u32_fn) {
                std::snprintf(status, status_size, "No u32 loader for %s.", desc.name);
                return false;
            }
            desc.load_u32_fn(
                buffers.x_train_u32,
                buffers.y_train_u8,
                buffers.x_test_u32,
                buffers.y_test_u8
            );
            std::snprintf(status, status_size, "Loaded %s (u32).", desc.name);
        } else {
            if (!desc.load_u8_fn) {
                std::snprintf(status, status_size, "No u8 loader for %s.", desc.name);
                return false;
            }
            desc.load_u8_fn(
                buffers.x_train_u8,
                buffers.y_train_u8,
                buffers.x_test_u8,
                buffers.y_test_u8
            );
            std::snprintf(status, status_size, "Loaded %s (u8).", desc.name);
        }
        return true;
    } catch (const std::exception& e) {
        std::snprintf(status, status_size, "Byte load failed: %s", e.what());
        return false;
    }
}

static bool LoadRelSprites(
    const kermac::RelSpritesConfig& cfg,
    DatasetBuffers& buffers,
    char* status,
    size_t status_size
) {
    try {
        kermac::relsprites_generate(cfg, buffers.x_train_f32, buffers.y_train_f32);
        std::snprintf(status, status_size, "Generated relsprites.");
        return true;
    } catch (const std::exception& e) {
        std::snprintf(status, status_size, "RelSprites generate failed: %s", e.what());
        return false;
    }
}

static bool LoadRelSpritesU32(
    const kermac::RelSpritesConfig& cfg,
    DatasetBuffers& buffers,
    char* status,
    size_t status_size
) {
    try {
        kermac::relsprites_generate_u32(cfg, buffers.x_train_u32, buffers.y_train_f32);
        std::snprintf(status, status_size, "Generated relsprites (u32).");
        return true;
    } catch (const std::exception& e) {
        std::snprintf(status, status_size, "RelSprites u32 generate failed: %s", e.what());
        return false;
    }
}
static int32_t LabelFromOneHot(
    const kermac::HostTensor<float>& labels,
    int64_t sample,
    int64_t num_labels
) {
    const float* ptr = labels.ptr();
    int64_t ld = labels.raw().stride[1];
    int32_t best = 0;
    float best_val = ptr[sample];
    for (int64_t i = 1; i < num_labels; ++i) {
        float v = ptr[sample + i * ld];
        if (v > best_val) {
            best_val = v;
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}

static int64_t GetSplitCount(const DatasetDims& dims, bool has_split, int split_idx) {
    if (!has_split) {
        return dims.num_train;
    }
    return (split_idx == 0) ? dims.num_train : dims.num_test;
}

static double BytesToMB(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static void DrawTableHeaderCell(const char* label, const char* tooltip) {
    ImGui::TableHeader(label);
    if (tooltip && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
        ImGui::SetTooltip("%s", tooltip);
    }
}

static bool LoadRelSpritesU8(
    const kermac::RelSpritesConfig& cfg,
    DatasetBuffers& buffers,
    char* status,
    size_t status_size
) {
    try {
        kermac::relsprites_generate_u8(cfg, buffers.x_train_u8, buffers.y_train_f32);
        std::snprintf(status, status_size, "Generated relsprites (u8).");
        return true;
    } catch (const std::exception& e) {
        std::snprintf(status, status_size, "RelSprites u8 generate failed: %s", e.what());
        return false;
    }
}

static void DrawTableCellTooltip(const char* tooltip) {
    if (ImGui::IsItemHovered()) {
        ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(ImGuiCol_HeaderHovered));
        if (tooltip && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
            ImGui::SetTooltip("%s", tooltip);
        }
    }
}

static void DrawTableCellText(const char* text, float row_height) {
    ImVec2 cell_pos = ImGui::GetCursorScreenPos();
    ImVec2 cell_size(ImGui::GetContentRegionAvail().x, row_height);
    ImGuiTable* table = ImGui::GetCurrentTable();
    if (table) {
        int column = ImGui::TableGetColumnIndex();
        ImRect rect = ImGui::TableGetCellBgRect(table, column);
        cell_pos = rect.Min;
        cell_size = rect.GetSize();
    }
    if (cell_size.x < 1.0f) {
        cell_size.x = 1.0f;
    }
    if (cell_size.y < 1.0f) {
        cell_size.y = row_height;
    }
    ImGui::SetCursorScreenPos(cell_pos);
    ImGui::InvisibleButton("##cell", cell_size);
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 text_pos = cell_pos;
    const ImGuiStyle& style = ImGui::GetStyle();
    text_pos.x += style.CellPadding.x;
    ImVec2 clip_min = cell_pos;
    ImVec2 clip_max(cell_pos.x + cell_size.x, cell_pos.y + cell_size.y);
    draw_list->PushClipRect(clip_min, clip_max, true);
    draw_list->AddText(text_pos, ImGui::GetColorU32(ImGuiCol_Text), text);
    draw_list->PopClipRect();
}

static float CalcColumnWidth(float min_width, const char* text) {
    if (!text || text[0] == '\0') {
        return min_width;
    }
    const ImGuiStyle& style = ImGui::GetStyle();
    float w = ImGui::CalcTextSize(text).x + style.CellPadding.x * 2.0f + style.FramePadding.x * 2.0f;
    return (w > min_width) ? w : min_width;
}

static void FormatRelSpritesValueTooltip(int label_idx, float value, char* out, size_t out_size) {
    if (!out || out_size == 0) {
        return;
    }
    out[0] = '\0';
    const int v_int = static_cast<int>(std::lrintf(value));
    switch (label_idx) {
        case 0: {
            if (v_int > 0) {
                std::snprintf(out, out_size, "B right of A by %d", v_int);
            } else if (v_int < 0) {
                std::snprintf(out, out_size, "B left of A by %d", -v_int);
            } else {
                std::snprintf(out, out_size, "B aligned with A in x");
            }
            break;
        }
        case 1: {
            if (v_int > 0) {
                std::snprintf(out, out_size, "B below A by %d", v_int);
            } else if (v_int < 0) {
                std::snprintf(out, out_size, "B above A by %d", -v_int);
            } else {
                std::snprintf(out, out_size, "B aligned with A in y");
            }
            break;
        }
        case 2:
            std::snprintf(out, out_size, "Distance between centers = %.3f", value);
            break;
        case 3:
            std::snprintf(out, out_size, "B is %s of A", (v_int >= 1) ? "right" : "left");
            break;
        case 4:
            std::snprintf(out, out_size, "B is %s A", (v_int >= 1) ? "below" : "above");
            break;
        case 5: {
            const char* quad = "left & above";
            switch (v_int) {
                case 1: quad = "right & above"; break;
                case 2: quad = "left & below"; break;
                case 3: quad = "right & below"; break;
                default: quad = "left & above"; break;
            }
            std::snprintf(out, out_size, "Quadrant: %s", quad);
            break;
        }
        case 6: {
            static const char* kAngle8Names[] = { "E", "NE", "N", "NW", "W", "SW", "S", "SE" };
            int idx = v_int & 7;
            std::snprintf(out, out_size, "Direction: %s", kAngle8Names[idx]);
            break;
        }
        case 7: {
            static const char* kDistBins[] = { "near", "mid", "far" };
            int idx = v_int;
            if (idx < 0) idx = 0;
            if (idx > 2) idx = 2;
            std::snprintf(out, out_size, "Distance bin: %s", kDistBins[idx]);
            break;
        }
        case 8:
            std::snprintf(out, out_size, "Circle radius rA = %d", v_int);
            break;
        case 9:
            std::snprintf(out, out_size, "Square half-size rB = %d", v_int);
            break;
        case 10:
            std::snprintf(out, out_size, "Circle center xA = %d", v_int);
            break;
        case 11:
            std::snprintf(out, out_size, "Circle center yA = %d", v_int);
            break;
        case 12:
            std::snprintf(out, out_size, "Square center xB = %d", v_int);
            break;
        case 13:
            std::snprintf(out, out_size, "Square center yB = %d", v_int);
            break;
        default:
            std::snprintf(out, out_size, "Value = %.3f", value);
            break;
    }
}

static void DrawLabelsTableClass(
    const DatasetDescriptor& dataset,
    const DatasetDims& dims,
    const kermac::HostTensor<float>* labels_f32,
    const kermac::HostTensor<uint8_t>* labels_u8,
    int64_t start_index,
    int64_t count
) {
    const int num_labels = static_cast<int>(dims.num_labels);
    const int column_count = 3 + num_labels;
    ImVec2 cell_padding = ImGui::GetStyle().CellPadding;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(cell_padding.x, 0.0f));
    ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX;
    if (!ImGui::BeginTable("LabelsTable", column_count, flags, ImVec2(0.0f, 0.0f))) {
        ImGui::PopStyleVar();
        return;
    }

    float index_width = CalcColumnWidth(60.0f, "Index");
    float label_width = CalcColumnWidth(60.0f, "Label");
    float name_width = CalcColumnWidth(120.0f, "Name");
    if (dataset.label_names) {
        for (int i = 0; i < dataset.num_label_names; ++i) {
            name_width = CalcColumnWidth(name_width, dataset.label_names[i]);
        }
    }

    ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed, index_width);
    ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, label_width);
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, name_width);
    for (int i = 0; i < num_labels; ++i) {
        char header[16];
        std::snprintf(header, sizeof(header), "oh%d", i);
        float col_width = CalcColumnWidth(50.0f, header);
        ImGui::TableSetupColumn(header, ImGuiTableColumnFlags_WidthFixed, col_width);
    }
    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
    ImGui::TableSetColumnIndex(0);
    DrawTableHeaderCell("Index", "Sample index");
    ImGui::TableSetColumnIndex(1);
    DrawTableHeaderCell("Label", "Integer class id (u8 if loaded, else argmax of one-hot)");
    ImGui::TableSetColumnIndex(2);
    DrawTableHeaderCell("Name", "Class name");
    for (int i = 0; i < num_labels; ++i) {
        ImGui::TableSetColumnIndex(3 + i);
        const char* name = (dataset.label_names && i < dataset.num_label_names)
            ? dataset.label_names[i]
            : nullptr;
        char header[16];
        std::snprintf(header, sizeof(header), "oh%d", i);
        char tooltip[128];
        if (name) {
            std::snprintf(tooltip, sizeof(tooltip), "%s, yes/no", name);
            DrawTableHeaderCell(header, tooltip);
        } else {
            DrawTableHeaderCell(header, "One-hot column (yes/no)");
        }
    }

    const float* f32_ptr = labels_f32 ? labels_f32->ptr() : nullptr;
    int64_t f32_ld = labels_f32 ? labels_f32->raw().stride[1] : 0;
    const uint8_t* u8_ptr = labels_u8 ? labels_u8->ptr() : nullptr;
    const float row_height = ImGui::GetTextLineHeight();

    for (int64_t row = 0; row < count; ++row) {
        int64_t sample = start_index + row;
        int32_t label_idx = -1;
        if (u8_ptr) {
            label_idx = static_cast<int32_t>(u8_ptr[sample]);
        } else if (f32_ptr) {
            label_idx = LabelFromOneHot(*labels_f32, sample, dims.num_labels);
        }

        const char* label_name = "unknown";
        if (label_idx >= 0 && label_idx < dataset.num_label_names) {
            label_name = dataset.label_names[label_idx];
        }

        ImGui::PushID(static_cast<int>(row));
        ImGui::TableNextRow(ImGuiTableRowFlags_None, row_height);
        ImGui::TableSetColumnIndex(0);
        ImGui::PushID(0);
        char cell_text[64];
        std::snprintf(cell_text, sizeof(cell_text), "%lld", static_cast<long long>(sample));
        DrawTableCellText(cell_text, row_height);
        DrawTableCellTooltip("Sample index");
        ImGui::PopID();
        ImGui::TableSetColumnIndex(1);
        ImGui::PushID(1);
        std::snprintf(cell_text, sizeof(cell_text), "%d", label_idx);
        DrawTableCellText(cell_text, row_height);
        DrawTableCellTooltip((label_idx >= 0 && label_idx < dataset.num_label_names)
            ? dataset.label_names[label_idx]
            : nullptr);
        ImGui::PopID();
        ImGui::TableSetColumnIndex(2);
        ImGui::PushID(2);
        DrawTableCellText(label_name, row_height);
        DrawTableCellTooltip(label_name);
        ImGui::PopID();
        for (int i = 0; i < num_labels; ++i) {
            float v = 0.0f;
            if (f32_ptr) {
                v = f32_ptr[sample + static_cast<int64_t>(i) * f32_ld];
            } else if (label_idx == i) {
                v = 1.0f;
            }
            ImGui::TableSetColumnIndex(3 + i);
            ImGui::PushID(3 + i);
            std::snprintf(cell_text, sizeof(cell_text), "%.1f", v);
            DrawTableCellText(cell_text, row_height);
            if (dataset.label_names && i < dataset.num_label_names) {
                const char* name = dataset.label_names[i];
                char tip[128];
                if (v >= 0.5f) {
                    std::snprintf(tip, sizeof(tip), "%s", name);
                } else {
                    std::snprintf(tip, sizeof(tip), "Not %s", name);
                }
                DrawTableCellTooltip(tip);
            } else {
                char tip[64];
                if (v >= 0.5f) {
                    std::snprintf(tip, sizeof(tip), "Class %d", i);
                } else {
                    std::snprintf(tip, sizeof(tip), "Not class %d", i);
                }
                DrawTableCellTooltip(tip);
            }
            ImGui::PopID();
        }
        ImGui::PopID();
    }

    ImGui::EndTable();
    ImGui::PopStyleVar();
}

static void DrawLabelsTableFields(
    const DatasetDescriptor& dataset,
    const DatasetDims& dims,
    const kermac::HostTensor<float>& labels,
    int64_t start_index,
    int64_t count
) {
    const int num_labels = static_cast<int>(dims.num_labels);
    const int column_count = 1 + num_labels;
    ImVec2 cell_padding = ImGui::GetStyle().CellPadding;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(cell_padding.x, 0.0f));
    ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX;
    if (!ImGui::BeginTable("LabelsTable", column_count, flags, ImVec2(0.0f, 0.0f))) {
        ImGui::PopStyleVar();
        return;
    }

    float index_width = CalcColumnWidth(60.0f, "Index");
    ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed, index_width);
    for (int i = 0; i < num_labels; ++i) {
        const char* name = (dataset.label_names && i < dataset.num_label_names)
            ? dataset.label_names[i]
            : nullptr;
        char header[32];
        if (name) {
            std::snprintf(header, sizeof(header), "%s", name);
        } else {
            std::snprintf(header, sizeof(header), "L%d", i);
        }
        float col_width = CalcColumnWidth(70.0f, header);
        ImGui::TableSetupColumn(header, ImGuiTableColumnFlags_WidthFixed, col_width);
    }
    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
    ImGui::TableSetColumnIndex(0);
    DrawTableHeaderCell("Index", "Sample index");
    for (int i = 0; i < num_labels; ++i) {
        ImGui::TableSetColumnIndex(1 + i);
        const char* name = (dataset.label_names && i < dataset.num_label_names)
            ? dataset.label_names[i]
            : nullptr;
        const char* desc = (dataset.label_desc && i < dataset.num_label_names)
            ? dataset.label_desc[i]
            : nullptr;
        char header[32];
        if (name) {
            std::snprintf(header, sizeof(header), "%s", name);
        } else {
            std::snprintf(header, sizeof(header), "L%d", i);
        }
        DrawTableHeaderCell(header, desc ? desc : "Field value");
    }

    const float* ptr = labels.ptr();
    int64_t ld = labels.raw().stride[1];
    const float row_height = ImGui::GetTextLineHeight();

    for (int64_t row = 0; row < count; ++row) {
        int64_t sample = start_index + row;
        ImGui::PushID(static_cast<int>(row));
        ImGui::TableNextRow(ImGuiTableRowFlags_None, row_height);
        ImGui::TableSetColumnIndex(0);
        ImGui::PushID(0);
        char cell_text[64];
        std::snprintf(cell_text, sizeof(cell_text), "%lld", static_cast<long long>(sample));
        DrawTableCellText(cell_text, row_height);
        DrawTableCellTooltip("Sample index");
        ImGui::PopID();
        for (int i = 0; i < num_labels; ++i) {
            float v = ptr[sample + static_cast<int64_t>(i) * ld];
            bool as_int = dataset.label_is_int && i < dataset.num_label_names && dataset.label_is_int[i];
            ImGui::TableSetColumnIndex(1 + i);
            ImGui::PushID(1 + i);
            if (as_int) {
                std::snprintf(cell_text, sizeof(cell_text), "%d", static_cast<int>(std::lrintf(v)));
                DrawTableCellText(cell_text, row_height);
            } else {
                std::snprintf(cell_text, sizeof(cell_text), "%.3f", v);
                DrawTableCellText(cell_text, row_height);
            }
            char tip[160];
            tip[0] = '\0';
            if (std::strcmp(dataset.name, "RelSprites") == 0) {
                FormatRelSpritesValueTooltip(i, v, tip, sizeof(tip));
            } else if (dataset.label_desc && i < dataset.num_label_names) {
                std::snprintf(tip, sizeof(tip), "%s", dataset.label_desc[i]);
            }
            DrawTableCellTooltip(tip[0] != '\0' ? tip : nullptr);
            ImGui::PopID();
        }
        ImGui::PopID();
    }

    ImGui::EndTable();
    ImGui::PopStyleVar();
}

static void DrawImageF32(
    const kermac::HostTensor<float>& images,
    int64_t sample,
    int64_t rows,
    int64_t cols,
    int64_t channels,
    float pixel_scale
) {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x < 1.0f || avail.y < 1.0f) {
        return;
    }

    float cell = pixel_scale > 0.0f ? pixel_scale : std::min(avail.x / cols, avail.y / rows);
    cell = std::max(1.0f, cell);
    float width = cell * cols;
    float height = cell * rows;

    ImVec2 origin = ImGui::GetCursorScreenPos();
    if (width < avail.x) {
        origin.x += 0.5f * (avail.x - width);
    }
    if (height < avail.y) {
        origin.y += 0.5f * (avail.y - height);
    }

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + height), IM_COL32(14, 14, 16, 255));

    const float* data = images.ptr();
    int64_t ld = images.raw().stride[1];
    int64_t plane = rows * cols;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            int64_t pixel_idx = r * cols + c;
            ImU32 color = IM_COL32(0, 0, 0, 255);
            if (channels <= 1) {
                int64_t idx = sample + pixel_idx * ld;
                float v = data[idx];
                v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                int grey = static_cast<int>(v * 255.0f + 0.5f);
                color = IM_COL32(grey, grey, grey, 255);
            } else {
                int64_t base = sample + pixel_idx * ld;
                float r_v = data[base + 0 * plane * ld];
                float g_v = data[base + 1 * plane * ld];
                float b_v = data[base + 2 * plane * ld];
                r_v = r_v < 0.0f ? 0.0f : (r_v > 1.0f ? 1.0f : r_v);
                g_v = g_v < 0.0f ? 0.0f : (g_v > 1.0f ? 1.0f : g_v);
                b_v = b_v < 0.0f ? 0.0f : (b_v > 1.0f ? 1.0f : b_v);
                int r_i = static_cast<int>(r_v * 255.0f + 0.5f);
                int g_i = static_cast<int>(g_v * 255.0f + 0.5f);
                int b_i = static_cast<int>(b_v * 255.0f + 0.5f);
                color = IM_COL32(r_i, g_i, b_i, 255);
            }
            ImVec2 p0(origin.x + c * cell, origin.y + r * cell);
            ImVec2 p1(p0.x + cell, p0.y + cell);
            draw_list->AddRectFilled(p0, p1, color);
        }
    }

    ImGui::Dummy(ImVec2(avail.x, std::max(avail.y, height)));
}

static void DrawImageGridF32(
    const kermac::HostTensor<float>& images,
    int64_t start_index,
    int64_t sample_count,
    int grid_rows,
    int grid_cols,
    int64_t rows,
    int64_t cols,
    int64_t channels,
    float pixel_scale
) {
    if (grid_rows <= 0 || grid_cols <= 0) {
        return;
    }
    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x < 1.0f || avail.y < 1.0f) {
        return;
    }

    const float padding = 6.0f;
    float cell = pixel_scale > 0.0f ? pixel_scale : 1.0f;
    float cell_w = cell * cols;
    float cell_h = cell * rows;
    float total_w = grid_cols * cell_w + (grid_cols - 1) * padding;
    float total_h = grid_rows * cell_h + (grid_rows - 1) * padding;

    if (total_w > avail.x || total_h > avail.y) {
        float scale_w = (avail.x - (grid_cols - 1) * padding) / (grid_cols * cols);
        float scale_h = (avail.y - (grid_rows - 1) * padding) / (grid_rows * rows);
        cell = std::max(1.0f, std::min(scale_w, scale_h));
        cell_w = cell * cols;
        cell_h = cell * rows;
        total_w = grid_cols * cell_w + (grid_cols - 1) * padding;
        total_h = grid_rows * cell_h + (grid_rows - 1) * padding;
    }

    ImVec2 origin = ImGui::GetCursorScreenPos();
    if (total_w < avail.x) {
        origin.x += 0.5f * (avail.x - total_w);
    }
    if (total_h < avail.y) {
        origin.y += 0.5f * (avail.y - total_h);
    }

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(origin, ImVec2(origin.x + total_w, origin.y + total_h), IM_COL32(12, 12, 14, 255));

    const float* data = images.ptr();
    int64_t ld = images.raw().stride[1];
    int64_t plane = rows * cols;

    int64_t max_samples = std::max<int64_t>(0, sample_count);
    for (int r = 0; r < grid_rows; ++r) {
        for (int c = 0; c < grid_cols; ++c) {
            int64_t sample = start_index + (r * grid_cols + c);
            if (sample >= max_samples) {
                continue;
            }
            ImVec2 img_origin(origin.x + c * (cell_w + padding), origin.y + r * (cell_h + padding));
            draw_list->AddRectFilled(img_origin, ImVec2(img_origin.x + cell_w, img_origin.y + cell_h), IM_COL32(14, 14, 16, 255));

            for (int64_t ir = 0; ir < rows; ++ir) {
                for (int64_t ic = 0; ic < cols; ++ic) {
                    int64_t pixel_idx = ir * cols + ic;
                    ImU32 color = IM_COL32(0, 0, 0, 255);
                    if (channels <= 1) {
                        int64_t idx = sample + pixel_idx * ld;
                        float v = data[idx];
                        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                        int grey = static_cast<int>(v * 255.0f + 0.5f);
                        color = IM_COL32(grey, grey, grey, 255);
                    } else {
                        int64_t base = sample + pixel_idx * ld;
                        float r_v = data[base + 0 * plane * ld];
                        float g_v = data[base + 1 * plane * ld];
                        float b_v = data[base + 2 * plane * ld];
                        r_v = r_v < 0.0f ? 0.0f : (r_v > 1.0f ? 1.0f : r_v);
                        g_v = g_v < 0.0f ? 0.0f : (g_v > 1.0f ? 1.0f : g_v);
                        b_v = b_v < 0.0f ? 0.0f : (b_v > 1.0f ? 1.0f : b_v);
                        int r_i = static_cast<int>(r_v * 255.0f + 0.5f);
                        int g_i = static_cast<int>(g_v * 255.0f + 0.5f);
                        int b_i = static_cast<int>(b_v * 255.0f + 0.5f);
                        color = IM_COL32(r_i, g_i, b_i, 255);
                    }
                    ImVec2 p0(img_origin.x + ic * cell, img_origin.y + ir * cell);
                    ImVec2 p1(p0.x + cell, p0.y + cell);
                    draw_list->AddRectFilled(p0, p1, color);
                }
            }
        }
    }

    ImGui::Dummy(ImVec2(avail.x, std::max(avail.y, total_h)));
}

static void DrawImageU8(
    const kermac::HostTensor<uint8_t>& images,
    int64_t sample,
    int64_t rows,
    int64_t cols,
    float pixel_scale
) {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x < 1.0f || avail.y < 1.0f) {
        return;
    }

    float cell = pixel_scale > 0.0f ? pixel_scale : std::min(avail.x / cols, avail.y / rows);
    cell = std::max(1.0f, cell);
    float width = cell * cols;
    float height = cell * rows;

    ImVec2 origin = ImGui::GetCursorScreenPos();
    if (width < avail.x) {
        origin.x += 0.5f * (avail.x - width);
    }
    if (height < avail.y) {
        origin.y += 0.5f * (avail.y - height);
    }

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + height), IM_COL32(14, 14, 16, 255));

    const uint8_t* data = images.ptr();
    int64_t ld = images.raw().stride[1];
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            int64_t idx = sample + (r * cols + c) * ld;
            uint8_t v = data[idx];
            ImU32 color = IM_COL32(v, v, v, 255);
            ImVec2 p0(origin.x + c * cell, origin.y + r * cell);
            ImVec2 p1(p0.x + cell, p0.y + cell);
            draw_list->AddRectFilled(p0, p1, color);
        }
    }

    ImGui::Dummy(ImVec2(avail.x, std::max(avail.y, height)));
}

static void DrawImageGridU8(
    const kermac::HostTensor<uint8_t>& images,
    int64_t start_index,
    int64_t sample_count,
    int grid_rows,
    int grid_cols,
    int64_t rows,
    int64_t cols,
    float pixel_scale
) {
    if (grid_rows <= 0 || grid_cols <= 0) {
        return;
    }
    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x < 1.0f || avail.y < 1.0f) {
        return;
    }

    const float padding = 6.0f;
    float cell = pixel_scale > 0.0f ? pixel_scale : 1.0f;
    float cell_w = cell * cols;
    float cell_h = cell * rows;
    float total_w = grid_cols * cell_w + (grid_cols - 1) * padding;
    float total_h = grid_rows * cell_h + (grid_rows - 1) * padding;

    if (total_w > avail.x || total_h > avail.y) {
        float scale_w = (avail.x - (grid_cols - 1) * padding) / (grid_cols * cols);
        float scale_h = (avail.y - (grid_rows - 1) * padding) / (grid_rows * rows);
        cell = std::max(1.0f, std::min(scale_w, scale_h));
        cell_w = cell * cols;
        cell_h = cell * rows;
        total_w = grid_cols * cell_w + (grid_cols - 1) * padding;
        total_h = grid_rows * cell_h + (grid_rows - 1) * padding;
    }

    ImVec2 origin = ImGui::GetCursorScreenPos();
    if (total_w < avail.x) {
        origin.x += 0.5f * (avail.x - total_w);
    }
    if (total_h < avail.y) {
        origin.y += 0.5f * (avail.y - total_h);
    }

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(origin, ImVec2(origin.x + total_w, origin.y + total_h), IM_COL32(12, 12, 14, 255));

    const uint8_t* data = images.ptr();
    int64_t ld = images.raw().stride[1];
    int64_t max_samples = std::max<int64_t>(0, sample_count);

    for (int r = 0; r < grid_rows; ++r) {
        for (int c = 0; c < grid_cols; ++c) {
            int64_t sample = start_index + (r * grid_cols + c);
            if (sample >= max_samples) {
                continue;
            }
            ImVec2 img_origin(origin.x + c * (cell_w + padding), origin.y + r * (cell_h + padding));
            draw_list->AddRectFilled(img_origin, ImVec2(img_origin.x + cell_w, img_origin.y + cell_h), IM_COL32(14, 14, 16, 255));

            for (int64_t ir = 0; ir < rows; ++ir) {
                for (int64_t ic = 0; ic < cols; ++ic) {
                    int64_t idx = sample + (ir * cols + ic) * ld;
                    uint8_t v = data[idx];
                    ImU32 color = IM_COL32(v, v, v, 255);
                    ImVec2 p0(img_origin.x + ic * cell, img_origin.y + ir * cell);
                    ImVec2 p1(p0.x + cell, p0.y + cell);
                    draw_list->AddRectFilled(p0, p1, color);
                }
            }
        }
    }

    ImGui::Dummy(ImVec2(avail.x, std::max(avail.y, total_h)));
}

static void DrawImageU32(
    const kermac::HostTensor<uint32_t>& images,
    int64_t sample,
    int64_t rows,
    int64_t cols,
    float pixel_scale
) {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x < 1.0f || avail.y < 1.0f) {
        return;
    }

    float cell = pixel_scale > 0.0f ? pixel_scale : std::min(avail.x / cols, avail.y / rows);
    cell = std::max(1.0f, cell);
    float width = cell * cols;
    float height = cell * rows;

    ImVec2 origin = ImGui::GetCursorScreenPos();
    if (width < avail.x) {
        origin.x += 0.5f * (avail.x - width);
    }
    if (height < avail.y) {
        origin.y += 0.5f * (avail.y - height);
    }

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + height), IM_COL32(14, 14, 16, 255));

    const uint32_t* data = images.ptr();
    int64_t ld = images.raw().stride[1];
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            int64_t idx = sample + (r * cols + c) * ld;
            uint32_t rgba = data[idx];
            uint8_t rr = (uint8_t)(rgba & 0xFFu);
            uint8_t gg = (uint8_t)((rgba >> 8) & 0xFFu);
            uint8_t bb = (uint8_t)((rgba >> 16) & 0xFFu);
            uint8_t aa = (uint8_t)((rgba >> 24) & 0xFFu);
            ImU32 color = IM_COL32(rr, gg, bb, aa);
            ImVec2 p0(origin.x + c * cell, origin.y + r * cell);
            ImVec2 p1(p0.x + cell, p0.y + cell);
            draw_list->AddRectFilled(p0, p1, color);
        }
    }

    ImGui::Dummy(ImVec2(avail.x, std::max(avail.y, height)));
}

static void DrawImageGridU32(
    const kermac::HostTensor<uint32_t>& images,
    int64_t start_index,
    int64_t sample_count,
    int grid_rows,
    int grid_cols,
    int64_t rows,
    int64_t cols,
    float pixel_scale
) {
    if (grid_rows <= 0 || grid_cols <= 0) {
        return;
    }
    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x < 1.0f || avail.y < 1.0f) {
        return;
    }

    const float padding = 6.0f;
    float cell = pixel_scale > 0.0f ? pixel_scale : 1.0f;
    float cell_w = cell * cols;
    float cell_h = cell * rows;
    float total_w = grid_cols * cell_w + (grid_cols - 1) * padding;
    float total_h = grid_rows * cell_h + (grid_rows - 1) * padding;

    if (total_w > avail.x || total_h > avail.y) {
        float scale_w = (avail.x - (grid_cols - 1) * padding) / (grid_cols * cols);
        float scale_h = (avail.y - (grid_rows - 1) * padding) / (grid_rows * rows);
        cell = std::max(1.0f, std::min(scale_w, scale_h));
        cell_w = cell * cols;
        cell_h = cell * rows;
        total_w = grid_cols * cell_w + (grid_cols - 1) * padding;
        total_h = grid_rows * cell_h + (grid_rows - 1) * padding;
    }

    ImVec2 origin = ImGui::GetCursorScreenPos();
    if (total_w < avail.x) {
        origin.x += 0.5f * (avail.x - total_w);
    }
    if (total_h < avail.y) {
        origin.y += 0.5f * (avail.y - total_h);
    }

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(origin, ImVec2(origin.x + total_w, origin.y + total_h), IM_COL32(12, 12, 14, 255));

    const uint32_t* data = images.ptr();
    int64_t ld = images.raw().stride[1];
    int64_t max_samples = std::max<int64_t>(0, sample_count);

    for (int r = 0; r < grid_rows; ++r) {
        for (int c = 0; c < grid_cols; ++c) {
            int64_t sample = start_index + (r * grid_cols + c);
            if (sample >= max_samples) {
                continue;
            }
            ImVec2 img_origin(origin.x + c * (cell_w + padding), origin.y + r * (cell_h + padding));
            draw_list->AddRectFilled(img_origin, ImVec2(img_origin.x + cell_w, img_origin.y + cell_h), IM_COL32(14, 14, 16, 255));

            for (int64_t ir = 0; ir < rows; ++ir) {
                for (int64_t ic = 0; ic < cols; ++ic) {
                    int64_t idx = sample + (ir * cols + ic) * ld;
                    uint32_t rgba = data[idx];
                    uint8_t rr = (uint8_t)(rgba & 0xFFu);
                    uint8_t gg = (uint8_t)((rgba >> 8) & 0xFFu);
                    uint8_t bb = (uint8_t)((rgba >> 16) & 0xFFu);
                    uint8_t aa = (uint8_t)((rgba >> 24) & 0xFFu);
                    ImU32 color = IM_COL32(rr, gg, bb, aa);
                    ImVec2 p0(img_origin.x + ic * cell, img_origin.y + ir * cell);
                    ImVec2 p1(p0.x + cell, p0.y + cell);
                    draw_list->AddRectFilled(p0, p1, color);
                }
            }
        }
    }

    ImGui::Dummy(ImVec2(avail.x, std::max(avail.y, total_h)));
}

static const char* kFmnistLabelNames[] = {
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
};

static const char* kMnistLabelNames[] = {
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
};

static const char* kCifar10LabelNames[] = {
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
};

static const char* kRelSpritesLabelNames[] = {
    "dx",
    "dy",
    "dist",
    "lr",
    "ud",
    "quad",
    "angle8",
    "dist_bin",
    "rA",
    "rB",
    "xA",
    "yA",
    "xB",
    "yB"
};

static const char* kRelSpritesLabelDesc[] = {
    "x offset of B relative to A (dx = xB - xA)",
    "y offset of B relative to A (dy = yB - yA, y grows downward)",
    "Euclidean distance between centers",
    "1 if B is right of A, else 0",
    "1 if B is below A, else 0",
    "Quadrant code (lr + 2*ud)",
    "Direction bin of A->B (0=E,1=NE,2=N,3=NW,4=W,5=SW,6=S,7=SE)",
    "Distance bin (0=near,1=mid,2=far)",
    "Circle radius (A)",
    "Square half-size (B)",
    "Circle center x (A)",
    "Circle center y (A)",
    "Square center x (B)",
    "Square center y (B)"
};

static const bool kRelSpritesLabelIsInt[] = {
    true,  // dx
    true,  // dy
    false, // dist
    true,  // lr
    true,  // ud
    true,  // quad
    true,  // angle8
    true,  // dist_bin
    true,  // rA
    true,  // rB
    true,  // xA
    true,  // yA
    true,  // xB
    true   // yB
};

static const int64_t kRelSpritesMaxN = 5000;
static const int64_t kRelSpritesMaxW = 64;
static const int64_t kRelSpritesMaxH = 64;

static const DatasetDescriptor kDatasets[] = {
    {
        "Fashion-MNIST",
        "10-class, 28x28 grayscale fashion images",
        60000,
        10000,
        28,
        28,
        1,
        kFmnistLabelNames,
        10,
        nullptr,
        nullptr,
        true,
        true,
        false,
        "u8 [0..255]",
        false,
        DatasetDescriptor::LabelKind::ClassIndex,
        &kermac::fmnist_dims,
        &kermac::fmnist_load,
        &kermac::fmnist_load_u8,
        nullptr
    },
    {
        "MNIST",
        "10-class, 28x28 handwritten digit images",
        60000,
        10000,
        28,
        28,
        1,
        kMnistLabelNames,
        10,
        nullptr,
        nullptr,
        true,
        true,
        false,
        "u8 [0..255]",
        false,
        DatasetDescriptor::LabelKind::ClassIndex,
        &kermac::mnist_dims,
        &kermac::mnist_load,
        &kermac::mnist_load_u8,
        nullptr
    },
    {
        "CIFAR-10",
        "10-class, 32x32 color images",
        50000,
        10000,
        32,
        32,
        3,
        kCifar10LabelNames,
        10,
        nullptr,
        nullptr,
        true,
        true,
        true,
        "u32 RGBA",
        false,
        DatasetDescriptor::LabelKind::ClassIndex,
        &kermac::cifar10_dims,
        &kermac::cifar10_load,
        nullptr,
        &kermac::cifar10_load_u32
    },
    {
        "RelSprites",
        "Synthetic relational sprites (circle vs square)",
        kRelSpritesMaxN,
        0,
        kRelSpritesMaxH,
        kRelSpritesMaxW,
        3,
        kRelSpritesLabelNames,
        14,
        kRelSpritesLabelDesc,
        kRelSpritesLabelIsInt,
        false,
        true,
        true,
        "u32 RGBA",
        true,
        DatasetDescriptor::LabelKind::FieldVector,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    }
};

static const char* DatasetComboGetter(void* data, int idx) {
    (void)data;
    const int count = static_cast<int>(IM_ARRAYSIZE(kDatasets));
    if (idx < 0 || idx >= count) {
        return nullptr;
    }
    return kDatasets[idx].name;
}

int main(int, char**) {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        return 1;
    }
    if (!glfwVulkanSupported()) {
        std::fprintf(stderr, "GLFW: Vulkan not supported\n");
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Kermac Data Viewer", nullptr, nullptr);

    ImVector<const char*> extensions;
    uint32_t extensions_count = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&extensions_count);
    for (uint32_t i = 0; i < extensions_count; ++i) {
        extensions.push_back(glfw_extensions[i]);
    }

    SetupVulkan(extensions);

    VkSurfaceKHR surface;
    VkResult err = glfwCreateWindowSurface(g_Instance, window, g_Allocator, &surface);
    check_vk_result(err);

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    ImGui_ImplVulkanH_Window* wd = &g_MainWindowData;
    SetupVulkanWindow(wd, surface, w, h);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = g_Instance;
    init_info.PhysicalDevice = g_PhysicalDevice;
    init_info.Device = g_Device;
    init_info.QueueFamily = g_QueueFamily;
    init_info.Queue = g_Queue;
    init_info.PipelineCache = g_PipelineCache;
    init_info.DescriptorPool = g_DescriptorPool;
    init_info.MinImageCount = g_MinImageCount;
    init_info.ImageCount = wd->ImageCount;
    init_info.Allocator = g_Allocator;
    init_info.PipelineInfoMain.RenderPass = wd->RenderPass;
    init_info.PipelineInfoMain.Subpass = 0;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.CheckVkResultFn = check_vk_result;

    ImGui_ImplVulkan_Init(&init_info);
    LoadFonts(window, io);

    constexpr int kDatasetCount = static_cast<int>(IM_ARRAYSIZE(kDatasets));
    static_assert(kDatasetCount == 4, "Update dataset buffer allocation when adding datasets.");

    RelSpritesState rel{};
    rel.cfg.w = 16;
    rel.cfg.h = 16;
    rel.cfg.n = 1000;
    rel.cfg.distractors = 4;
    rel.cfg.noise = 0.05f;
    rel.cfg.seed = 1u;
    rel.cfg.mode = kermac::RelSpritesMode::KERMAC_RELSPRITES_MODE_GRAY;

    size_t host_bytes = 0;
    void* host_memory_ptr = nullptr;

    ViewerState view;
    ActiveStorageSlot storage;
    bool images_loaded = false;
    bool f32_labels_loaded = false;
    bool u8_labels_loaded = false;
    char status[256] = {};
    size_t active_bytes = 0;

    auto rebuild_storage = [&](const DatasetDescriptor& dataset, const DatasetDims& dims, bool use_byte, bool use_u32_bytes) {
        BufferPlan plan = MakeBufferPlan(dataset, dims, use_byte, use_u32_bytes);
        active_bytes = MeasureBuffers(dims, plan);
        storage.reset();
        if (host_memory_ptr) {
            kermac::host_free(host_memory_ptr);
            host_memory_ptr = nullptr;
        }
        host_bytes = active_bytes;
        host_memory_ptr = kermac::host_alloc(host_bytes);
        storage.emplace(host_memory_ptr, host_bytes, dims, plan);
        images_loaded = false;
        f32_labels_loaded = false;
        u8_labels_loaded = false;
        status[0] = '\0';
    };

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        int fb_width = 0;
        int fb_height = 0;
        glfwGetFramebufferSize(window, &fb_width, &fb_height);
        if (fb_width > 0 && fb_height > 0 &&
            (g_SwapChainRebuild || wd->Width != fb_width || wd->Height != fb_height)) {
            ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
            ImGui_ImplVulkanH_CreateOrResizeWindow(
                g_Instance,
                g_PhysicalDevice,
                g_Device,
                wd,
                g_QueueFamily,
                g_Allocator,
                fb_width,
                fb_height,
                g_MinImageCount,
                0
            );
            wd->FrameIndex = 0;
            g_SwapChainRebuild = false;
        }
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Data Viewer");
        bool dataset_changed = false;
        if (ImGui::Combo("Dataset", &view.dataset_idx, DatasetComboGetter, nullptr, kDatasetCount)) {
            dataset_changed = true;
            view.sample_idx = 0;
        }

        const DatasetDescriptor& dataset = kDatasets[view.dataset_idx];
        DatasetDims dims = dataset.supports_params
            ? RelSpritesDimsFromConfig(rel.cfg)
            : QueryDatasetDims(dataset);

        bool params_changed = false;

        if (dataset.supports_params) {
            ImGui::Separator();
            ImGui::Text("RelSprites parameters");

            int n = rel.cfg.n;
            if (ImGui::InputInt("Samples (N)", &n)) {
                n = std::max(1, std::min(n, static_cast<int>(kRelSpritesMaxN)));
                if (n != rel.cfg.n) {
                    rel.cfg.n = n;
                    params_changed = true;
                }
            }

            int w_cfg = rel.cfg.w;
            if (ImGui::InputInt("Width (W)", &w_cfg)) {
                w_cfg = std::max(8, std::min(w_cfg, static_cast<int>(kRelSpritesMaxW)));
                if (w_cfg != rel.cfg.w) {
                    rel.cfg.w = w_cfg;
                    params_changed = true;
                }
            }

            int h_cfg = rel.cfg.h;
            if (ImGui::InputInt("Height (H)", &h_cfg)) {
                h_cfg = std::max(8, std::min(h_cfg, static_cast<int>(kRelSpritesMaxH)));
                if (h_cfg != rel.cfg.h) {
                    rel.cfg.h = h_cfg;
                    params_changed = true;
                }
            }

            int distractors = rel.cfg.distractors;
            if (ImGui::InputInt("Distractors", &distractors)) {
                distractors = std::max(0, std::min(distractors, 32));
                if (distractors != rel.cfg.distractors) {
                    rel.cfg.distractors = distractors;
                    params_changed = true;
                }
            }

            float noise = rel.cfg.noise;
            if (ImGui::InputFloat("Noise", &noise, 0.01f, 0.05f, "%.3f")) {
                noise = std::max(0.0f, std::min(noise, 1.0f));
                if (noise != rel.cfg.noise) {
                    rel.cfg.noise = noise;
                    params_changed = true;
                }
            }

            int seed = static_cast<int>(rel.cfg.seed);
            if (ImGui::InputInt("Seed", &seed)) {
                if (seed <= 0) seed = 1;
                if (static_cast<uint32_t>(seed) != rel.cfg.seed) {
                    rel.cfg.seed = static_cast<uint32_t>(seed);
                    params_changed = true;
                }
            }

            const char* mode_labels[] = { "binary", "gray", "color" };
            int mode_idx = 0;
            switch (rel.cfg.mode) {
                case kermac::RelSpritesMode::KERMAC_RELSPRITES_MODE_GRAY:
                    mode_idx = 1;
                    break;
                case kermac::RelSpritesMode::KERMAC_RELSPRITES_MODE_COLOR:
                    mode_idx = 2;
                    break;
                default:
                    mode_idx = 0;
                    break;
            }
            if (ImGui::Combo("Mode", &mode_idx, mode_labels, IM_ARRAYSIZE(mode_labels))) {
                rel.cfg.mode = (mode_idx == 2)
                    ? kermac::RelSpritesMode::KERMAC_RELSPRITES_MODE_COLOR
                    : (mode_idx == 1)
                        ? kermac::RelSpritesMode::KERMAC_RELSPRITES_MODE_GRAY
                        : kermac::RelSpritesMode::KERMAC_RELSPRITES_MODE_BINARY;
                params_changed = true;
            }
        }

        if (params_changed) {
            dims = RelSpritesDimsFromConfig(rel.cfg);
        }

        bool type_changed = false;
        const char* byte_label = dataset.byte_label;
        if (dataset.supports_params && dims.image_channels == 1) {
            byte_label = "u8 [0..255]";
        }

        if (dataset.supports_byte) {
            const char* type_labels[] = { "float [0..1]", byte_label };
            if (ImGui::Combo("Type", &view.type_idx, type_labels, IM_ARRAYSIZE(type_labels))) {
                type_changed = true;
            }
        } else {
            if (view.type_idx != 0) {
                view.type_idx = 0;
                type_changed = true;
            }
            ImGui::Text("Type: float [0..1]");
        }

        const bool use_byte = dataset.supports_byte && view.type_idx == 1;
        bool use_u32_bytes = use_byte && dataset.byte_is_u32;
        if (dataset.supports_params && dims.image_channels == 1) {
            use_u32_bytes = false;
        }

        const bool rebuild_needed = dataset_changed || type_changed || params_changed || !storage.valid;
        if (rebuild_needed) {
            rebuild_storage(dataset, dims, use_byte, use_u32_bytes);
            view.sample_idx = 0;
        }

        DatasetBuffers& active_buffers = storage.get()->buffers;
        char* status_active = status;

        if (use_byte) {
            if (!images_loaded) {
                if (dataset.supports_params) {
                    if (use_u32_bytes) {
                        images_loaded = LoadRelSpritesU32(rel.cfg, active_buffers, status_active, sizeof(status));
                    } else {
                        images_loaded = LoadRelSpritesU8(rel.cfg, active_buffers, status_active, sizeof(status));
                    }
                    f32_labels_loaded = images_loaded;
                } else {
                    images_loaded = LoadDatasetBytes(dataset, active_buffers, status_active, sizeof(status));
                    u8_labels_loaded = images_loaded;
                }
            }
        } else {
            if (!images_loaded) {
                images_loaded = dataset.supports_params
                    ? LoadRelSprites(rel.cfg, active_buffers, status_active, sizeof(status))
                    : LoadDatasetF32(dataset, active_buffers, status_active, sizeof(status));
                f32_labels_loaded = images_loaded;
            }
        }

        ImGui::Text("%s", dataset.name);
        ImGui::Text("%s", dataset.summary);
        ImGui::Separator();
        ImGui::Text("Train samples: %lld", static_cast<long long>(dims.num_train));
        ImGui::Text("Test samples:  %lld", static_cast<long long>(dims.num_test));
        ImGui::Text("Image: %lld x %lld x %lld (%lld dims)",
                    static_cast<long long>(dims.image_rows),
                    static_cast<long long>(dims.image_cols),
                    static_cast<long long>(dims.image_channels),
                    static_cast<long long>(dims.num_dims));
        ImGui::Text("Labels: %lld", static_cast<long long>(dims.num_labels));
        ImGui::Text("Active bytes: %zu (%.2f MB)", active_bytes, BytesToMB(active_bytes));
        ImGui::Text("Host capacity: %zu (%.2f MB)", host_bytes, BytesToMB(host_bytes));
        ImGui::Separator();

        if (dataset.has_split) {
            const char* split_labels[] = { "train", "test" };
            if (ImGui::Combo("Split", &view.split_idx, split_labels, IM_ARRAYSIZE(split_labels))) {
                view.sample_idx = 0;
            }
        } else {
            view.split_idx = 0;
            ImGui::Text("Split: all");
        }

        int64_t split_count = GetSplitCount(dims, dataset.has_split, view.split_idx);
        int max_index = static_cast<int>(split_count) - 1;
        if (max_index < 0) {
            max_index = 0;
        }
        view.sample_idx = std::max(0, std::min(view.sample_idx, max_index));
        ImGui::SliderInt("Start index", &view.sample_idx, 0, max_index);
        if (ImGui::InputInt("Start index (Exact)", &view.sample_idx)) {
            view.sample_idx = std::max(0, std::min(view.sample_idx, max_index));
        }
        const int kMontageMaxDim = 16;
        if (ImGui::InputInt("Montage rows", &view.montage_rows)) {
            view.montage_rows = std::max(1, std::min(view.montage_rows, kMontageMaxDim));
        }
        if (ImGui::InputInt("Montage cols", &view.montage_cols)) {
            view.montage_cols = std::max(1, std::min(view.montage_cols, kMontageMaxDim));
        }
        ImGui::InputFloat("Pixel scale", &view.pixel_scale, 1.0f, 4.0f, "%.1f");
        if (view.pixel_scale < 1.0f) {
            view.pixel_scale = 1.0f;
        }

        int64_t montage_count = static_cast<int64_t>(view.montage_rows) * static_cast<int64_t>(view.montage_cols);
        if (montage_count < 1) {
            montage_count = 1;
        }
        int64_t max_rows = split_count - view.sample_idx;
        if (max_rows < 0) {
            max_rows = 0;
        }
        int64_t table_count = std::min(montage_count, max_rows);

        ImGui::Separator();
        ImGui::Text("Images loaded (%s): %s", use_byte ? byte_label : "float [0..1]", images_loaded ? "yes" : "no");
        if (status_active[0] != '\0') {
            ImGui::Text("Status: %s", status_active);
        }
        ImGui::End();

        ImGui::Begin("Labels");
        ImGui::Text("%s", dataset.name);
        ImGui::Text("Rows: %lld (start %d)", static_cast<long long>(table_count), view.sample_idx);
        ImGui::Separator();
        ImGui::Text("Labels: %lld", static_cast<long long>(dims.num_labels));
        ImGui::Text("Float labels loaded: %s", f32_labels_loaded ? "yes" : "no");
        if (dataset.label_kind == DatasetDescriptor::LabelKind::ClassIndex) {
            ImGui::Text("U8 labels loaded: %s", u8_labels_loaded ? "yes" : "no");
        }
        ImGui::Separator();

        if (dataset.label_kind == DatasetDescriptor::LabelKind::FieldVector) {
            if (f32_labels_loaded) {
                const kermac::HostTensor<float>& labels =
                    (view.split_idx == 0) ? active_buffers.y_train_f32 : active_buffers.y_test_f32;
                DrawLabelsTableFields(dataset, dims, labels, view.sample_idx, table_count);
            } else {
                ImGui::Text("Labels: (not loaded)");
            }
        } else {
            const kermac::HostTensor<float>* labels_f32 =
                f32_labels_loaded
                    ? ((view.split_idx == 0) ? &active_buffers.y_train_f32 : &active_buffers.y_test_f32)
                    : nullptr;
            const kermac::HostTensor<uint8_t>* labels_u8 =
                u8_labels_loaded
                    ? ((view.split_idx == 0) ? &active_buffers.y_train_u8 : &active_buffers.y_test_u8)
                    : nullptr;
            if (labels_f32 || labels_u8) {
                DrawLabelsTableClass(dataset, dims, labels_f32, labels_u8, view.sample_idx, table_count);
            } else {
                ImGui::Text("Labels: (not loaded)");
            }
        }
        ImGui::End();

        ImGui::Begin("Image");
        ImGui::Text("%s", dataset.name);
        ImGui::Text("Montage: %dx%d (start %d)", view.montage_rows, view.montage_cols, view.sample_idx);
        ImGui::Separator();
        if (!use_byte && images_loaded) {
            const kermac::HostTensor<float>& images =
                (view.split_idx == 0) ? active_buffers.x_train_f32 : active_buffers.x_test_f32;
            DrawImageGridF32(
                images,
                view.sample_idx,
                split_count,
                view.montage_rows,
                view.montage_cols,
                dims.image_rows,
                dims.image_cols,
                dims.image_channels,
                view.pixel_scale
            );
        } else if (use_byte && images_loaded) {
            if (use_u32_bytes) {
                const kermac::HostTensor<uint32_t>& images =
                    (view.split_idx == 0) ? active_buffers.x_train_u32 : active_buffers.x_test_u32;
                DrawImageGridU32(
                    images,
                    view.sample_idx,
                    split_count,
                    view.montage_rows,
                    view.montage_cols,
                    dims.image_rows,
                    dims.image_cols,
                    view.pixel_scale
                );
            } else {
                const kermac::HostTensor<uint8_t>& images =
                    (view.split_idx == 0) ? active_buffers.x_train_u8 : active_buffers.x_test_u8;
                DrawImageGridU8(
                    images,
                    view.sample_idx,
                    split_count,
                    view.montage_rows,
                    view.montage_cols,
                    dims.image_rows,
                    dims.image_cols,
                    view.pixel_scale
                );
            }
        } else {
            ImGui::Text("Dataset not loaded.");
        }
        ImGui::End();

        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        const bool minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
        if (!minimized) {
            wd->ClearValue.color.float32[0] = 0.05f;
            wd->ClearValue.color.float32[1] = 0.05f;
            wd->ClearValue.color.float32[2] = 0.06f;
            wd->ClearValue.color.float32[3] = 1.0f;
            FrameRender(wd, draw_data);
            FramePresent(wd);
        }
    }

    storage.reset();
    if (host_memory_ptr) {
        kermac::host_free(host_memory_ptr);
        host_memory_ptr = nullptr;
        host_bytes = 0;
    }

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    CleanupVulkanWindow();
    CleanupVulkan();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
