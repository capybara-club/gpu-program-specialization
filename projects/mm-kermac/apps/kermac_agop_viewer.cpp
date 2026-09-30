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
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include <kermac.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

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

struct ComputeConfig {
    int n = 128;
    int d = 8;
    int seed = 1234;
    int iterations = 1;
    float bandwidth = 10.0f;
    float reg = 1e-3f;
    enum class KernelPath {
        SEMIRING_COPY_GRAD = 0,
        LEGACY_COPY_GRAD
    };
    KernelPath kernel_path = KernelPath::SEMIRING_COPY_GRAD;
    kermac::AgopBackend backend = kermac::AgopBackend::KERMAC_AGOP_BACKEND_CUTENSOR_F32;
    kermac::AgopOutput output = kermac::AgopOutput::KERMAC_AGOP_OUTPUT_FULL;
};

struct ComputeResults {
    int n = 0;
    int d = 0;
    bool output_diag = false;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> feature;
    float x0_min = 0.0f;
    float x0_max = 0.0f;
    float x1_min = 0.0f;
    float x1_max = 0.0f;
    float y_min = 0.0f;
    float y_max = 0.0f;
    float feature_min = 0.0f;
    float feature_max = 0.0f;
    float symmetry_max_diff = 0.0f;
    int factor_info = 0;
    int solve_info = 0;
    size_t feature_nan = 0;
    size_t feature_inf = 0;
    size_t host_bytes = 0;
    size_t device_bytes = 0;
    double compute_ms = 0.0;
    int iterations = 0;
    std::string status;
    bool ok = false;
};

struct HostMemDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            kermac::host_free(ptr);
        }
    }
};

struct VizConfig {
    int scatter_stride = 1;
    float point_radius = 2.2f;
    bool heatmap_signed = false;
    bool heatmap_auto = true;
    float heatmap_scale = 1.0f;
};

struct GpuContext {
    std::unique_ptr<kermac::Kermac> handle;
    std::unique_ptr<kermac::Agop> agop;
    std::unique_ptr<kermac::LegacyHandle> legacy;
    std::unique_ptr<kermac::Semiring> semiring_copy_grad;
    size_t agop_scratch_bytes = 0;
    size_t semiring_copy_grad_scratch_bytes = 0;
    float semiring_copy_grad_bandwidth = 0.0f;
    float semiring_copy_grad_regularizer = 0.0f;
    float semiring_copy_grad_epsilon = 0.0f;
    kermac::MatrixPackedType semiring_copy_grad_packed =
        kermac::MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE;
    CUstream stream = nullptr;
    std::string error;
    bool ok = false;

    GpuContext() {
        if (cuInit(0) != CUDA_SUCCESS) {
            error = "cuInit failed";
            return;
        }
        int count = 0;
        if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count < 1) {
            error = "No CUDA devices found";
            return;
        }
        try {
            handle = std::make_unique<kermac::Kermac>(2);
        } catch (const std::exception& e) {
            error = e.what();
            return;
        }
        if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
            error = "cuStreamCreate failed";
            handle.reset();
            return;
        }
        ok = true;
    }

    ~GpuContext() {
        if (stream != nullptr) {
            (void)cuStreamDestroy(stream);
            stream = nullptr;
        }
    }
};

static bool EnsureAgop(
    GpuContext& gpu,
    size_t scratch_bytes,
    std::string& error
) {
    if (!gpu.handle) {
        error = "GPU handle not initialized";
        return false;
    }
    if (gpu.ok == false) {
        error = gpu.error;
        return false;
    }
    if (scratch_bytes < (1u << 20)) {
        scratch_bytes = 1u << 20;
    }
    const size_t max_scratch = 1u << 28;
    for (; scratch_bytes <= max_scratch; scratch_bytes <<= 1) {
        void* scratch_ptr = nullptr;
        try {
            scratch_ptr = kermac::host_alloc(scratch_bytes);
        } catch (const std::exception& e) {
            error = "agop scratch host_alloc failed (bytes=" +
                std::to_string(scratch_bytes) + "): " + e.what();
            return false;
        }
        std::unique_ptr<void, HostMemDeleter> scratch_mem(scratch_ptr);
        kermac::HostStackAllocator hsa_scratch(scratch_mem.get(), scratch_bytes);
        try {
            kermac::Agop tmp = kermac::Agop::create(*gpu.handle, hsa_scratch);
            gpu.agop = std::make_unique<kermac::Agop>(std::move(tmp));
            gpu.agop_scratch_bytes = scratch_bytes;
            return true;
        } catch (const kermac::Error& e) {
            if (e.result() != KERMAC_ERROR_OUT_OF_MEMORY) {
                error = e.what();
                return false;
            }
        } catch (const std::exception& e) {
            error = e.what();
            return false;
        }
    }
    error = "agop scratch allocation too small (exceeded " + std::to_string(max_scratch) + " bytes)";
    return false;
}

static bool EnsureLegacy(
    GpuContext& gpu,
    std::string& error
) {
    if (!gpu.handle) {
        error = "GPU handle not initialized";
        return false;
    }
    if (gpu.ok == false) {
        error = gpu.error;
        return false;
    }
    if (gpu.legacy) {
        return true;
    }
    try {
        gpu.legacy = std::make_unique<kermac::LegacyHandle>(*gpu.handle);
    } catch (const kermac::Error& e) {
        error = e.what();
        return false;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
    return true;
}

static bool EnsureSemiringCopyGrad(
    GpuContext& gpu,
    size_t scratch_bytes,
    kermac::MatrixPackedType packed_type,
    float bandwidth,
    float regularizer,
    float epsilon,
    std::string& error
) {
    if (!gpu.handle) {
        error = "GPU handle not initialized";
        return false;
    }
    if (gpu.ok == false) {
        error = gpu.error;
        return false;
    }
    if (gpu.semiring_copy_grad &&
        gpu.semiring_copy_grad_packed == packed_type &&
        gpu.semiring_copy_grad_bandwidth == bandwidth &&
        gpu.semiring_copy_grad_regularizer == regularizer &&
        gpu.semiring_copy_grad_epsilon == epsilon) {
        return true;
    }

    gpu.semiring_copy_grad.reset();
    gpu.semiring_copy_grad_packed = packed_type;
    gpu.semiring_copy_grad_bandwidth = bandwidth;
    gpu.semiring_copy_grad_regularizer = regularizer;
    gpu.semiring_copy_grad_epsilon = epsilon;

    if (scratch_bytes < (1u << 20)) {
        scratch_bytes = 1u << 20;
    }
    const size_t max_scratch = 1u << 28;
    for (; scratch_bytes <= max_scratch; scratch_bytes <<= 1) {
        void* scratch_ptr = nullptr;
        try {
            scratch_ptr = kermac::host_alloc(scratch_bytes);
        } catch (const std::exception& e) {
            error = "semiring scratch host_alloc failed (bytes=" +
                std::to_string(scratch_bytes) + "): " + e.what();
            return false;
        }
        std::unique_ptr<void, HostMemDeleter> scratch_mem(scratch_ptr);
        kermac::HostStackAllocator hsa_scratch(scratch_mem.get(), scratch_bytes);
        try {
            kermac::Semiring tmp = kermac::Semiring::laplace_l2_symm_copy_grad(
                *gpu.handle,
                hsa_scratch,
                packed_type,
                bandwidth,
                regularizer,
                epsilon
            );
            gpu.semiring_copy_grad = std::make_unique<kermac::Semiring>(std::move(tmp));
            gpu.semiring_copy_grad_scratch_bytes = scratch_bytes;
            return true;
        } catch (const kermac::Error& e) {
            if (e.result() != KERMAC_ERROR_OUT_OF_MEMORY) {
                error = e.what();
                return false;
            }
        } catch (const std::exception& e) {
            error = e.what();
            return false;
        }
    }

    error = "semiring scratch allocation too small (exceeded " +
        std::to_string(max_scratch) + " bytes)";
    return false;
}

static ImU32 LerpColor(const ImVec4& a, const ImVec4& b, float t) {
    t = (t < 0.0f) ? 0.0f : (t > 1.0f) ? 1.0f : t;
    ImVec4 c(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
        a.w + (b.w - a.w) * t
    );
    return ImGui::ColorConvertFloat4ToU32(c);
}

static void FillDataTensor(
    kermac::HostTensor<float>& dst,
    const std::vector<float>& src,
    int n,
    int d
) {
    const kermac::TensorRaw raw = dst.raw();
    const int64_t ld = raw.stride[1];
    float* ptr = dst.ptr();
    for (int col = 0; col < d; ++col) {
        for (int row = 0; row < n; ++row) {
            ptr[col * ld + row] = src[row * d + col];
        }
    }
}

static void FillVectorTensor(
    kermac::HostTensor<float>& dst,
    const std::vector<float>& src,
    int n
) {
    float* ptr = dst.ptr();
    for (int i = 0; i < n; ++i) {
        ptr[i] = src[i];
    }
}

static bool ComputeKernelMatricesDevice(
    const ComputeConfig& cfg,
    const bool allow_semiring_copy_grad,
    GpuContext& gpu,
    kermac::DeviceStackAllocator& dsa,
    kermac::DeviceTensor<float>& data_metric,
    kermac::DeviceTensor<float>& data_base,
    kermac::DeviceTensor<float>& kernel_train,
    kermac::DeviceTensor<float>& kernel_grad,
    float epsilon,
    std::string& error
) {
    const auto packed_type = kermac::MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE;
    if (!EnsureLegacy(gpu, error)) {
        return false;
    }

    try {
        if (cfg.kernel_path == ComputeConfig::KernelPath::SEMIRING_COPY_GRAD && allow_semiring_copy_grad) {
            if (!EnsureSemiringCopyGrad(
                    gpu,
                    gpu.semiring_copy_grad_scratch_bytes > 0 ? gpu.semiring_copy_grad_scratch_bytes : (1u << 20),
                    packed_type,
                    cfg.bandwidth,
                    cfg.reg,
                    epsilon,
                    error)) {
                return false;
            }
            // Semiring copy-grad path computes both kernel values and grad terms in one pass.
            // It only supports the identity metric currently.
            gpu.semiring_copy_grad->run(data_base, data_base, kernel_train, gpu.stream);
        } else {
            const kermac::TensorCoreMode tcm = kermac::TensorCoreMode::KERMAC_TENSOR_CORE_MODE_F32;
            kermac::contraction(
                *gpu.handle,
                dsa,
                tcm,
                1.0f,
                data_metric, "mk",
                data_base, "nk",
                0.0f,
                kernel_train, "mn",
                kernel_train, "mn",
                gpu.stream
            );
            kermac::laplace_symmetric(
                *gpu.legacy,
                packed_type,
                dsa,
                kernel_train,
                cfg.bandwidth,
                cfg.reg,
                epsilon,
                gpu.stream
            );
        }

        kernel_grad.copy_from(kernel_train, gpu.stream);
        kermac::copy_triangle(
            *gpu.legacy,
            kermac::MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE,
            kernel_grad,
            0.0f,
            gpu.stream
        );
    } catch (const kermac::Error& e) {
        error = e.what();
        return false;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }

    return true;
}

static bool ComputeFeatureMatrix(
    const ComputeConfig& cfg,
    GpuContext& gpu,
    ComputeResults& out
) {
    out = ComputeResults{};
    if (!gpu.ok) {
        out.status = gpu.error;
        return false;
    }
    if (cfg.n < 4 || cfg.d < 2) {
        out.status = "n or d too small";
        return false;
    }
    const int n = cfg.n;
    const int d = cfg.d;
    const int iterations = std::max(1, cfg.iterations);
    const int mask_idx = (d > 5) ? 5 : (d - 1);
    const bool output_diag = (cfg.output == kermac::AgopOutput::KERMAC_AGOP_OUTPUT_DIAG);
    const float epsilon = 1e-6f;

    std::vector<float> x(static_cast<size_t>(n) * d);
    std::vector<float> y(static_cast<size_t>(n));
    std::mt19937 rng(cfg.seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < d; ++k) {
            x[i * d + k] = dist(rng);
        }
        float mask = x[i * d + mask_idx] > 0.0f ? 1.0f : 0.0f;
        y[i] = x[i * d + 0] * x[i * d + 1] * mask;
    }

    float x0_min = x[0];
    float x0_max = x[0];
    float x1_min = x[1];
    float x1_max = x[1];
    float y_min = y[0];
    float y_max = y[0];
    for (int i = 0; i < n; ++i) {
        float x0 = x[i * d + 0];
        float x1 = x[i * d + 1];
        x0_min = std::min(x0_min, x0);
        x0_max = std::max(x0_max, x0);
        x1_min = std::min(x1_min, x1);
        x1_max = std::max(x1_max, x1);
        y_min = std::min(y_min, y[i]);
        y_max = std::max(y_max, y[i]);
    }

    size_t host_bytes = 0;
    size_t host_info_bytes = 0;
    {
        kermac::HostStackAllocator hsa_dry(nullptr, 0);
        kermac::HostTensor<float> h_data_base(hsa_dry, n, d);
        kermac::HostTensor<float> h_solution_rhs(hsa_dry, n, 1);
        kermac::HostTensor<float> h_metric(hsa_dry, d, d);
        kermac::HostTensor<float> h_feature_full(hsa_dry, d, d);
        (void)h_data_base;
        (void)h_solution_rhs;
        (void)h_metric;
        (void)h_feature_full;
        host_bytes = hsa_dry.get().largest_total_offset;
    }

    {
        kermac::HostStackAllocator hsa_info_dry(nullptr, 0);
        kermac::HostTensor<int32_t> h_factor_info(hsa_info_dry, 1);
        kermac::HostTensor<int32_t> h_solve_info(hsa_info_dry, 1);
        (void)h_factor_info;
        (void)h_solve_info;
        host_info_bytes = hsa_info_dry.get().largest_total_offset;
    }

    out.host_bytes = host_bytes + host_info_bytes;
    void* host_ptr = kermac::host_alloc(host_bytes);
    std::unique_ptr<void, HostMemDeleter> host_mem(host_ptr);

    kermac::HostStackAllocator hsa(host_mem.get(), host_bytes);
    kermac::HostTensor<float> h_data_base(hsa, n, d);
    kermac::HostTensor<float> h_solution_rhs(hsa, n, 1);
    kermac::HostTensor<float> h_metric(hsa, d, d);
    kermac::HostTensor<float> h_feature_full(hsa, d, d);
    {
        const kermac::TensorRaw metric_raw = h_metric.raw();
        const int64_t ld = metric_raw.stride[1];
        float* metric_ptr = h_metric.ptr();
        for (int col = 0; col < d; ++col) {
            for (int row = 0; row < d; ++row) {
                metric_ptr[col * ld + row] = (row == col) ? 1.0f : 0.0f;
            }
        }
    }
    void* host_info_ptr = host_info_bytes ? kermac::host_alloc(host_info_bytes) : nullptr;
    std::unique_ptr<void, HostMemDeleter> host_info_mem(host_info_ptr);
    kermac::HostStackAllocator hsa_info(host_info_mem.get(), host_info_bytes);
    kermac::HostTensor<int32_t> h_factor_info(hsa_info, 1);
    kermac::HostTensor<int32_t> h_solve_info(hsa_info, 1);
    if (cfg.backend == kermac::AgopBackend::KERMAC_AGOP_BACKEND_FUSED && !gpu.agop) {
        if (!EnsureAgop(gpu, gpu.agop_scratch_bytes > 0 ? gpu.agop_scratch_bytes : (1u << 20), out.status)) {
            return false;
        }
    }
    if (!EnsureLegacy(gpu, out.status)) {
        return false;
    }
    auto run_agop = [&](kermac::DeviceStackAllocator& dsa,
                        kermac::AgopOutput output_mode,
                        kermac::DeviceTensor<float>& kernel_grad,
                        kermac::DeviceTensor<float>& data_n,
                        kermac::DeviceTensor<float>& solution,
                        kermac::DeviceTensor<float>& data_m,
                        kermac::DeviceTensor<float>& feature_out) {
        KermacAgop raw = gpu.agop ? gpu.agop->raw() : KermacAgop{};
        KERMAC_THROW_IF_ERROR(
            ::kermac_agop_run(
                gpu.handle->get(),
                dsa.get_ptr(),
                raw,
                cfg.backend,
                output_mode,
                cfg.bandwidth,
                kernel_grad.raw(),
                data_n.raw(),
                solution.raw(),
                data_m.raw(),
                feature_out.raw(),
                gpu.stream
            )
        );
    };

    FillDataTensor(h_data_base, x, n, d);
    FillVectorTensor(h_solution_rhs, y, n);

    size_t device_bytes = 0;
    {
        kermac::DeviceStackAllocator dsa_dry(nullptr, 0);
        kermac::DeviceTensor<float> d_data_base(dsa_dry, n, d);
        kermac::DeviceTensor<float> d_data_metric(dsa_dry, n, d);
        kermac::DeviceTensor<float> d_metric(dsa_dry, d, d);
        kermac::DeviceTensor<float> d_kernel_train(dsa_dry, n, n);
        kermac::DeviceTensor<float> d_kernel_grad(dsa_dry, n, n);
        kermac::DeviceTensor<float> d_solution(dsa_dry, n, 1);
        kermac::DeviceTensor<int32_t> d_factor_info(dsa_dry, 1);
        kermac::DeviceTensor<int32_t> d_solve_info(dsa_dry, 1);
        kermac::DeviceTensor<float> d_feature_full(dsa_dry, d, d);

        kermac::contraction(
            *gpu.handle,
            dsa_dry,
            kermac::TensorCoreMode::KERMAC_TENSOR_CORE_MODE_F32,
            1.0f,
            d_data_base, "mk",
            d_metric, "kn",
            0.0f,
            d_data_metric, "mn",
            d_data_metric, "mn",
            gpu.stream
        );

        if (!ComputeKernelMatricesDevice(
                cfg,
                true,
                gpu,
                dsa_dry,
                d_data_metric,
                d_data_base,
                d_kernel_train,
                d_kernel_grad,
                epsilon,
                out.status)) {
            return false;
        }
        if (cfg.kernel_path == ComputeConfig::KernelPath::SEMIRING_COPY_GRAD && iterations > 1) {
            if (!ComputeKernelMatricesDevice(
                    cfg,
                    false,
                    gpu,
                    dsa_dry,
                    d_data_metric,
                    d_data_base,
                    d_kernel_train,
                    d_kernel_grad,
                    epsilon,
                    out.status)) {
                return false;
            }
        }
        kermac::solve(
            *gpu.handle,
            kermac::MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE,
            dsa_dry,
            d_kernel_train,
            d_solution,
            d_factor_info,
            d_solve_info,
            gpu.stream
        );
        run_agop(
            dsa_dry,
            kermac::AgopOutput::KERMAC_AGOP_OUTPUT_FULL,
            d_kernel_grad,
            d_data_metric,
            d_solution,
            d_data_metric,
            d_feature_full
        );
        device_bytes = dsa_dry.get().largest_total_offset;
    }

    struct DeviceMemDeleter {
        kermac::Kermac* handle = nullptr;
        void operator()(void* ptr) const {
            if (ptr && handle) {
                kermac::device_free(*handle, ptr);
            }
        }
    };

    DeviceMemDeleter device_deleter{gpu.handle.get()};
    out.device_bytes = device_bytes;
    std::fprintf(stderr, "[agop_viewer] device_alloc bytes=%zu\n", device_bytes);
    void* device_ptr = kermac::device_alloc(*gpu.handle, device_bytes);
    std::fprintf(stderr, "[agop_viewer] device_alloc ok ptr=%p\n", device_ptr);
    std::unique_ptr<void, DeviceMemDeleter> device_mem(device_ptr, device_deleter);
    kermac::DeviceStackAllocator dsa(device_mem.get(), device_bytes);

    kermac::DeviceTensor<float> d_data_base(dsa, n, d);
    kermac::DeviceTensor<float> d_data_metric(dsa, n, d);
    kermac::DeviceTensor<float> d_metric(dsa, d, d);
    kermac::DeviceTensor<float> d_kernel_train(dsa, n, n);
    kermac::DeviceTensor<float> d_kernel_grad(dsa, n, n);
    kermac::DeviceTensor<float> d_solution(dsa, n, 1);
    kermac::DeviceTensor<int32_t> d_factor_info(dsa, 1);
    kermac::DeviceTensor<int32_t> d_solve_info(dsa, 1);
    kermac::DeviceTensor<float> d_feature_full(dsa, d, d);

    d_data_base.copy_from(h_data_base, gpu.stream);
    d_metric.copy_from(h_metric, gpu.stream);
    d_solution.copy_from(h_solution_rhs, gpu.stream);
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        kermac::contraction(
            *gpu.handle,
            dsa,
            kermac::TensorCoreMode::KERMAC_TENSOR_CORE_MODE_F32,
            1.0f,
            d_data_base, "mk",
            d_metric, "kn",
            0.0f,
            d_data_metric, "mn",
            d_data_metric, "mn",
            gpu.stream
        );

        if (!ComputeKernelMatricesDevice(
                cfg,
                iter == 0,
                gpu,
                dsa,
                d_data_metric,
                d_data_base,
                d_kernel_train,
                d_kernel_grad,
                epsilon,
                out.status)) {
            out.status = "iteration " + std::to_string(iter + 1) + ": " + out.status;
            return false;
        }

        // Solve always uses the original target vector y as RHS for each iteration.
        d_solution.copy_from(h_solution_rhs, gpu.stream);
        kermac::solve(
            *gpu.handle,
            kermac::MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE,
            dsa,
            d_kernel_train,
            d_solution,
            d_factor_info,
            d_solve_info,
            gpu.stream
        );
        h_factor_info.copy_from(d_factor_info, gpu.stream);
        h_solve_info.copy_from(d_solve_info, gpu.stream);
        if (cuStreamSynchronize(gpu.stream) != CUDA_SUCCESS) {
            out.status = "iteration " + std::to_string(iter + 1) + ": CUDA sync failed after solve";
            return false;
        }
        out.factor_info = h_factor_info.ptr()[0];
        out.solve_info = h_solve_info.ptr()[0];
        if (out.factor_info != 0 || out.solve_info != 0) {
            out.status = "iteration " + std::to_string(iter + 1) +
                ": solve failed (factor_info=" + std::to_string(out.factor_info) +
                ", solve_info=" + std::to_string(out.solve_info) + ")";
            return false;
        }

        run_agop(
            dsa,
            kermac::AgopOutput::KERMAC_AGOP_OUTPUT_FULL,
            d_kernel_grad,
            d_data_metric,
            d_solution,
            d_data_metric,
            d_feature_full
        );
        if (iter + 1 < iterations) {
            d_metric.copy_from(d_feature_full, gpu.stream);
        }
    }

    h_feature_full.copy_from(d_feature_full, gpu.stream);
    if (cuStreamSynchronize(gpu.stream) != CUDA_SUCCESS) {
        out.status = "CUDA sync failed after AGOP";
        return false;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end - start;

    std::vector<float> feature;
    float feature_min = 0.0f;
    float feature_max = 0.0f;
    float symmetry_max = 0.0f;
    size_t nan_count = 0;
    size_t inf_count = 0;
    if (output_diag) {
        feature.resize(static_cast<size_t>(d));
        const kermac::TensorRaw raw = h_feature_full.raw();
        const int64_t ld = raw.stride[1];
        const float* ptr = h_feature_full.ptr();
        for (int i = 0; i < d; ++i) {
            feature[i] = ptr[i * ld + i];
        }
        feature_min = feature[0];
        feature_max = feature[0];
        for (int i = 0; i < d; ++i) {
            float v = feature[i];
            if (std::isnan(v)) {
                nan_count++;
                continue;
            }
            if (!std::isfinite(v)) {
                inf_count++;
                continue;
            }
            feature_min = std::min(feature_min, v);
            feature_max = std::max(feature_max, v);
        }
    } else {
        feature.resize(static_cast<size_t>(d) * d);
        const kermac::TensorRaw raw = h_feature_full.raw();
        const int64_t ld = raw.stride[1];
        const float* ptr = h_feature_full.ptr();
        for (int col = 0; col < d; ++col) {
            for (int row = 0; row < d; ++row) {
                feature[col * d + row] = ptr[col * ld + row];
            }
        }
        feature_min = feature[0];
        feature_max = feature[0];
        for (size_t i = 0; i < feature.size(); ++i) {
            float v = feature[i];
            if (std::isnan(v)) {
                nan_count++;
                continue;
            }
            if (!std::isfinite(v)) {
                inf_count++;
                continue;
            }
            feature_min = std::min(feature_min, v);
            feature_max = std::max(feature_max, v);
        }
        for (int col = 0; col < d; ++col) {
            for (int row = 0; row < d; ++row) {
                float a = feature[col * d + row];
                float b = feature[row * d + col];
                symmetry_max = std::max(symmetry_max, std::fabs(a - b));
            }
        }
    }

    out.n = n;
    out.d = d;
    out.output_diag = output_diag;
    out.x = std::move(x);
    out.y = std::move(y);
    out.feature = std::move(feature);
    out.x0_min = x0_min;
    out.x0_max = x0_max;
    out.x1_min = x1_min;
    out.x1_max = x1_max;
    out.y_min = y_min;
    out.y_max = y_max;
    out.feature_min = feature_min;
    out.feature_max = feature_max;
    out.symmetry_max_diff = symmetry_max;
    out.feature_nan = nan_count;
    out.feature_inf = inf_count;
    out.host_bytes = host_bytes + host_info_bytes;
    out.device_bytes = device_bytes;
    out.compute_ms = elapsed_ms.count();
    out.iterations = iterations;
    out.status = "ok";
    out.ok = true;
    return true;
}

static bool EstimateMemoryUsage(
    const ComputeConfig& cfg,
    GpuContext& gpu,
    size_t& host_bytes_out,
    size_t& device_bytes_out,
    std::string& error
) {
    host_bytes_out = 0;
    device_bytes_out = 0;
    error.clear();

    if (!gpu.ok) {
        error = gpu.error;
        return false;
    }
    if (cfg.n < 4 || cfg.d < 2) {
        error = "n or d too small";
        return false;
    }

    const int n = cfg.n;
    const int d = cfg.d;
    const int iterations = std::max(1, cfg.iterations);
    const float epsilon = 1e-6f;

    size_t host_data_bytes = 0;
    size_t host_info_bytes = 0;
    {
        kermac::HostStackAllocator hsa_dry(nullptr, 0);
        kermac::HostTensor<float> h_data_base(hsa_dry, n, d);
        kermac::HostTensor<float> h_solution_rhs(hsa_dry, n, 1);
        kermac::HostTensor<float> h_metric(hsa_dry, d, d);
        kermac::HostTensor<float> h_feature_full(hsa_dry, d, d);
        (void)h_data_base;
        (void)h_solution_rhs;
        (void)h_metric;
        (void)h_feature_full;
        host_data_bytes = hsa_dry.get().largest_total_offset;
    }
    {
        kermac::HostStackAllocator hsa_info_dry(nullptr, 0);
        kermac::HostTensor<int32_t> h_factor_info(hsa_info_dry, 1);
        kermac::HostTensor<int32_t> h_solve_info(hsa_info_dry, 1);
        (void)h_factor_info;
        (void)h_solve_info;
        host_info_bytes = hsa_info_dry.get().largest_total_offset;
    }
    host_bytes_out = host_data_bytes + host_info_bytes;

    if (cfg.backend == kermac::AgopBackend::KERMAC_AGOP_BACKEND_FUSED && !gpu.agop) {
        if (!EnsureAgop(gpu, gpu.agop_scratch_bytes > 0 ? gpu.agop_scratch_bytes : (1u << 20), error)) {
            return false;
        }
    }
    if (!EnsureLegacy(gpu, error)) {
        return false;
    }

    auto run_agop = [&](kermac::DeviceStackAllocator& dsa,
                        kermac::DeviceTensor<float>& kernel_grad,
                        kermac::DeviceTensor<float>& data_n,
                        kermac::DeviceTensor<float>& solution,
                        kermac::DeviceTensor<float>& data_m,
                        kermac::DeviceTensor<float>& feature_out) {
        KermacAgop raw = gpu.agop ? gpu.agop->raw() : KermacAgop{};
        KERMAC_THROW_IF_ERROR(
            ::kermac_agop_run(
                gpu.handle->get(),
                dsa.get_ptr(),
                raw,
                cfg.backend,
                kermac::AgopOutput::KERMAC_AGOP_OUTPUT_FULL,
                cfg.bandwidth,
                kernel_grad.raw(),
                data_n.raw(),
                solution.raw(),
                data_m.raw(),
                feature_out.raw(),
                gpu.stream
            )
        );
    };

    try {
        kermac::DeviceStackAllocator dsa_dry(nullptr, 0);
        kermac::DeviceTensor<float> d_data_base(dsa_dry, n, d);
        kermac::DeviceTensor<float> d_data_metric(dsa_dry, n, d);
        kermac::DeviceTensor<float> d_metric(dsa_dry, d, d);
        kermac::DeviceTensor<float> d_kernel_train(dsa_dry, n, n);
        kermac::DeviceTensor<float> d_kernel_grad(dsa_dry, n, n);
        kermac::DeviceTensor<float> d_solution(dsa_dry, n, 1);
        kermac::DeviceTensor<int32_t> d_factor_info(dsa_dry, 1);
        kermac::DeviceTensor<int32_t> d_solve_info(dsa_dry, 1);
        kermac::DeviceTensor<float> d_feature_full(dsa_dry, d, d);

        kermac::contraction(
            *gpu.handle,
            dsa_dry,
            kermac::TensorCoreMode::KERMAC_TENSOR_CORE_MODE_F32,
            1.0f,
            d_data_base, "mk",
            d_metric, "kn",
            0.0f,
            d_data_metric, "mn",
            d_data_metric, "mn",
            gpu.stream
        );

        if (!ComputeKernelMatricesDevice(
                cfg,
                true,
                gpu,
                dsa_dry,
                d_data_metric,
                d_data_base,
                d_kernel_train,
                d_kernel_grad,
                epsilon,
                error)) {
            return false;
        }
        if (cfg.kernel_path == ComputeConfig::KernelPath::SEMIRING_COPY_GRAD && iterations > 1) {
            if (!ComputeKernelMatricesDevice(
                    cfg,
                    false,
                    gpu,
                    dsa_dry,
                    d_data_metric,
                    d_data_base,
                    d_kernel_train,
                    d_kernel_grad,
                    epsilon,
                    error)) {
                return false;
            }
        }

        kermac::solve(
            *gpu.handle,
            kermac::MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE,
            dsa_dry,
            d_kernel_train,
            d_solution,
            d_factor_info,
            d_solve_info,
            gpu.stream
        );

        run_agop(
            dsa_dry,
            d_kernel_grad,
            d_data_metric,
            d_solution,
            d_data_metric,
            d_feature_full
        );

        device_bytes_out = dsa_dry.get().largest_total_offset;
    } catch (const kermac::Error& e) {
        error = e.what();
        return false;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }

    return true;
}

static void DrawScatterPlot(const ComputeResults& data, const VizConfig& viz) {
    ImVec2 canvas = ImGui::GetContentRegionAvail();
    if (canvas.x < 10.0f || canvas.y < 10.0f) {
        return;
    }
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 max_pt(origin.x + canvas.x, origin.y + canvas.y);
    draw_list->AddRectFilled(origin, max_pt, IM_COL32(18, 20, 26, 255));
    draw_list->AddRect(origin, max_pt, IM_COL32(80, 80, 90, 255));

    float x_span = (data.x0_max - data.x0_min);
    float y_span = (data.x1_max - data.x1_min);
    if (x_span <= 1e-6f) {
        x_span = 1.0f;
    }
    if (y_span <= 1e-6f) {
        y_span = 1.0f;
    }
    float y_abs_max = std::max(std::fabs(data.y_min), std::fabs(data.y_max));
    if (y_abs_max <= 1e-6f) {
        y_abs_max = 1.0f;
    }

    const ImVec4 neg_color(0.2f, 0.45f, 0.95f, 1.0f);
    const ImVec4 pos_color(0.95f, 0.35f, 0.25f, 1.0f);
    const ImVec4 neutral(0.85f, 0.85f, 0.9f, 1.0f);

    const int stride = std::max(1, viz.scatter_stride);
    for (int i = 0; i < data.n; i += stride) {
        float x0 = data.x[i * data.d + 0];
        float x1 = data.x[i * data.d + 1];
        float y = data.y[i];
        float nx = (x0 - data.x0_min) / x_span;
        float ny = (x1 - data.x1_min) / y_span;
        ImVec2 p(
            origin.x + nx * canvas.x,
            origin.y + (1.0f - ny) * canvas.y
        );
        float t = y / y_abs_max;
        ImU32 color = (t >= 0.0f)
            ? LerpColor(neutral, pos_color, t)
            : LerpColor(neutral, neg_color, -t);
        draw_list->AddCircleFilled(p, viz.point_radius, color, 8);
    }

    ImGui::Dummy(canvas);
}

static void DrawHeatmap(const ComputeResults& data, const VizConfig& viz) {
    ImVec2 canvas = ImGui::GetContentRegionAvail();
    if (canvas.x < 10.0f || canvas.y < 10.0f) {
        return;
    }
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 max_pt(origin.x + canvas.x, origin.y + canvas.y);
    draw_list->AddRectFilled(origin, max_pt, IM_COL32(16, 18, 24, 255));
    draw_list->AddRect(origin, max_pt, IM_COL32(80, 80, 90, 255));

    const ImVec4 low_color(0.18f, 0.25f, 0.7f, 1.0f);
    const ImVec4 high_color(0.95f, 0.45f, 0.2f, 1.0f);
    const ImVec4 mid_color(0.9f, 0.9f, 0.9f, 1.0f);

    float scale = 1.0f;
    if (viz.heatmap_signed) {
        float abs_max = std::max(std::fabs(data.feature_min), std::fabs(data.feature_max));
        if (viz.heatmap_auto) {
            scale = (abs_max > 1e-6f) ? abs_max : 1.0f;
        } else {
            scale = (viz.heatmap_scale > 1e-6f) ? viz.heatmap_scale : 1.0f;
        }
    } else {
        float span = data.feature_max - data.feature_min;
        if (viz.heatmap_auto) {
            scale = (span > 1e-6f) ? span : 1.0f;
        } else {
            scale = (viz.heatmap_scale > 1e-6f) ? viz.heatmap_scale : 1.0f;
        }
    }

    const int d = data.d;
    if (d < 1) {
        ImGui::Dummy(canvas);
        return;
    }

    float cell_w = canvas.x / d;
    float cell_h = canvas.y / d;
    for (int col = 0; col < d; ++col) {
        for (int row = 0; row < d; ++row) {
            float v = data.feature[col * d + row];
            ImU32 color;
            if (viz.heatmap_signed) {
                float t = v / scale;
                if (t >= 0.0f) {
                    color = LerpColor(mid_color, high_color, t);
                } else {
                    color = LerpColor(mid_color, low_color, -t);
                }
            } else {
                float t = (v - data.feature_min) / scale;
                color = LerpColor(low_color, high_color, t);
            }
            ImVec2 p0(origin.x + col * cell_w, origin.y + row * cell_h);
            ImVec2 p1(origin.x + (col + 1) * cell_w, origin.y + (row + 1) * cell_h);
            draw_list->AddRectFilled(p0, p1, color);
        }
    }

    ImGui::Dummy(canvas);
}

static void DrawDiagPlot(const ComputeResults& data) {
    ImVec2 canvas = ImGui::GetContentRegionAvail();
    if (canvas.x < 10.0f || canvas.y < 10.0f || data.feature.empty()) {
        return;
    }
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 max_pt(origin.x + canvas.x, origin.y + canvas.y);
    draw_list->AddRectFilled(origin, max_pt, IM_COL32(18, 20, 26, 255));
    draw_list->AddRect(origin, max_pt, IM_COL32(80, 80, 90, 255));

    float min_v = data.feature_min;
    float max_v = data.feature_max;
    float span = max_v - min_v;
    if (span <= 1e-6f) {
        span = 1.0f;
    }
    float bar_w = canvas.x / data.d;

    for (int i = 0; i < data.d; ++i) {
        float v = data.feature[i];
        float t = (v - min_v) / span;
        float h = t * canvas.y;
        ImVec2 p0(origin.x + i * bar_w, origin.y + (canvas.y - h));
        ImVec2 p1(origin.x + (i + 1) * bar_w, origin.y + canvas.y);
        draw_list->AddRectFilled(p0, p1, IM_COL32(230, 140, 60, 255));
    }

    ImGui::Dummy(canvas);
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
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Kermac AGOP Viewer", nullptr, nullptr);

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
    //init_info.ApiVersion = VK_API_VERSION_1_3;
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

    GpuContext gpu;
    ComputeConfig cfg;
    ComputeResults results;
    VizConfig viz;
    bool auto_compute = true;
    size_t estimated_host_bytes = 0;
    size_t estimated_device_bytes = 0;
    std::string estimate_status;
    bool estimate_ok = false;
    bool estimate_dirty = true;
    bool pending_auto_compute = false;

    if (gpu.ok) {
        (void)ComputeFeatureMatrix(cfg, gpu, results);
    } else {
        results.status = gpu.error;
    }

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

        bool request_compute = false;
        bool edit_finished = false;
        bool params_changed = false;
        ImGui::Begin("AGOP Controls");
        ImGui::Text("F* dataset and AGOP feature matrix");
        ImGui::Separator();

        if (ImGui::InputInt("Seed", &cfg.seed)) {
            params_changed = true;
        }
        edit_finished = edit_finished || ImGui::IsItemDeactivatedAfterEdit();
        if (ImGui::InputInt("Samples (N)", &cfg.n)) {
            cfg.n = std::max(16, std::min(cfg.n, 4096));
            params_changed = true;
        }
        edit_finished = edit_finished || ImGui::IsItemDeactivatedAfterEdit();
        if (ImGui::InputInt("Features (D)", &cfg.d)) {
            cfg.d = std::max(6, std::min(cfg.d, 64));
            params_changed = true;
        }
        edit_finished = edit_finished || ImGui::IsItemDeactivatedAfterEdit();
        if (ImGui::InputInt("RFM/AGOP iterations", &cfg.iterations)) {
            cfg.iterations = std::max(1, std::min(cfg.iterations, 1024));
            params_changed = true;
        }
        edit_finished = edit_finished || ImGui::IsItemDeactivatedAfterEdit();
        if (ImGui::InputFloat("Bandwidth", &cfg.bandwidth, 0.1f, 1.0f, "%.3f")) {
            cfg.bandwidth = std::max(0.5f, std::min(cfg.bandwidth, 20.0f));
            params_changed = true;
        }
        edit_finished = edit_finished || ImGui::IsItemDeactivatedAfterEdit();
        if (ImGui::InputFloat("Regularizer", &cfg.reg, 1e-6f, 1e-4f, "%.6f", ImGuiInputTextFlags_CharsScientific)) {
            cfg.reg = std::max(1e-6f, std::min(cfg.reg, 1e-1f));
            params_changed = true;
        }
        edit_finished = edit_finished || ImGui::IsItemDeactivatedAfterEdit();

        const char* kernel_labels[] = {
            "semiring_copy_grad",
            "legacy_copy_grad"
        };
        int kernel_idx = 0;
        if (cfg.kernel_path == ComputeConfig::KernelPath::LEGACY_COPY_GRAD) {
            kernel_idx = 1;
        }
        if (ImGui::Combo("Kernel path", &kernel_idx, kernel_labels, IM_ARRAYSIZE(kernel_labels))) {
            cfg.kernel_path = (kernel_idx == 1)
                ? ComputeConfig::KernelPath::LEGACY_COPY_GRAD
                : ComputeConfig::KernelPath::SEMIRING_COPY_GRAD;
            request_compute = auto_compute;
            params_changed = true;
        }

        const char* backend_labels[] = {
            "cutensor_f32",
            "cutensor_tf32",
            "fused_semiring"
        };
        int backend_idx = 0;
        switch (cfg.backend) {
            case kermac::AgopBackend::KERMAC_AGOP_BACKEND_CUTENSOR_TF32:
                backend_idx = 1;
                break;
            case kermac::AgopBackend::KERMAC_AGOP_BACKEND_FUSED:
                backend_idx = 2;
                break;
            default:
                backend_idx = 0;
                break;
        }
        if (ImGui::Combo("Backend", &backend_idx, backend_labels, IM_ARRAYSIZE(backend_labels))) {
            if (backend_idx == 1) {
                cfg.backend = kermac::AgopBackend::KERMAC_AGOP_BACKEND_CUTENSOR_TF32;
            } else if (backend_idx == 2) {
                cfg.backend = kermac::AgopBackend::KERMAC_AGOP_BACKEND_FUSED;
            } else {
                cfg.backend = kermac::AgopBackend::KERMAC_AGOP_BACKEND_CUTENSOR_F32;
            }
            request_compute = auto_compute;
            params_changed = true;
        }

        const char* output_labels[] = { "full (D x D)", "diag (D)" };
        int output_idx = cfg.output == kermac::AgopOutput::KERMAC_AGOP_OUTPUT_DIAG ? 1 : 0;
        if (ImGui::Combo("Output", &output_idx, output_labels, IM_ARRAYSIZE(output_labels))) {
            cfg.output = (output_idx == 1)
                ? kermac::AgopOutput::KERMAC_AGOP_OUTPUT_DIAG
                : kermac::AgopOutput::KERMAC_AGOP_OUTPUT_FULL;
            request_compute = auto_compute;
            params_changed = true;
        }

        ImGui::Checkbox("Auto recompute", &auto_compute);
        if (ImGui::Button("Recompute")) {
            request_compute = true;
        }

        if (params_changed) {
            estimate_dirty = true;
            pending_auto_compute = true;
        }
        if (auto_compute && (edit_finished || (pending_auto_compute && !ImGui::IsAnyItemActive()))) {
            request_compute = true;
        }
        if (estimate_dirty) {
            estimate_ok = EstimateMemoryUsage(
                cfg,
                gpu,
                estimated_host_bytes,
                estimated_device_bytes,
                estimate_status
            );
            estimate_dirty = false;
        }

        ImGui::Separator();
        ImGui::Text("Host bytes (cfg): %zu", estimate_ok ? estimated_host_bytes : results.host_bytes);
        ImGui::Text("Device bytes (cfg): %zu", estimate_ok ? estimated_device_bytes : results.device_bytes);
        if (estimate_ok == false && !estimate_status.empty()) {
            ImGui::Text("Memory estimate: %s", estimate_status.c_str());
        }
        ImGui::Text("Host bytes (last run): %zu", results.host_bytes);
        ImGui::Text("Device bytes (last run): %zu", results.device_bytes);
        ImGui::Text("Iterations: %d", results.iterations);
        ImGui::Text("Compute time: %.2f ms", results.compute_ms);
        if (!results.output_diag) {
            ImGui::Text("Symmetry max diff: %.3e", results.symmetry_max_diff);
        }
        ImGui::Text("Factor info: %d", results.factor_info);
        ImGui::Text("Solve info: %d", results.solve_info);
        ImGui::Text("Feature NaN/Inf: %zu/%zu", results.feature_nan, results.feature_inf);
        ImGui::Text("Status: %s", results.status.c_str());

        ImGui::Separator();
        ImGui::Text("Visual controls");
        if (ImGui::InputInt("Scatter stride", &viz.scatter_stride)) {
            viz.scatter_stride = std::max(1, std::min(viz.scatter_stride, 64));
        }
        if (ImGui::InputFloat("Point radius", &viz.point_radius, 0.1f, 0.5f, "%.2f")) {
            viz.point_radius = std::max(0.5f, std::min(viz.point_radius, 8.0f));
        }
        ImGui::Checkbox("Signed heatmap", &viz.heatmap_signed);
        ImGui::Checkbox("Auto heat scale", &viz.heatmap_auto);
        if (!viz.heatmap_auto) {
            if (ImGui::InputFloat("Heat scale", &viz.heatmap_scale, 0.1f, 1.0f, "%.3f")) {
                viz.heatmap_scale = std::max(0.001f, std::min(viz.heatmap_scale, 100.0f));
            }
        }
        ImGui::End();

        if (request_compute) {
            pending_auto_compute = false;
            (void)ComputeFeatureMatrix(cfg, gpu, results);
            estimate_dirty = true;
        }

        ImGui::Begin("F* Dataset");
        if (results.ok) {
            DrawScatterPlot(results, viz);
        } else {
            ImGui::Text("No dataset available.");
        }
        ImGui::End();

        ImGui::Begin("Feature Matrix");
        if (results.ok) {
            if (results.output_diag) {
                DrawDiagPlot(results);
            } else {
                DrawHeatmap(results, viz);
            }
        } else {
            ImGui::Text("No feature matrix available.");
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

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    CleanupVulkanWindow();
    CleanupVulkan();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
