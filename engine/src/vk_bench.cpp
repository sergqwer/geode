// vk_bench.cpp - the GPU benchmark over Vulkan compute: what the GPU is, which data types its shaders can use, and
// how fast it runs FP32 / FP16 / INT32 arithmetic and reads memory. The engine itself does not use the GPU yet; this
// is the data for deciding whether it should.
#include "bench.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "bench_json.hpp"
#include "platform.hpp"

#if defined(__ANDROID__)
#include <vulkan/vulkan.h>
#define GEODE_VULKAN 1
// Vulkan structs are set up as {sType} and filled field by field
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

namespace geode::bench {

#if defined(GEODE_VULKAN)

namespace {

#include "vk_shaders.inc"

struct Gpu {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem{};

    ~Gpu() {
        if (pool) vkDestroyCommandPool(device, pool, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

struct Buffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void *map = nullptr;
    VkDeviceSize size = 0;

    bool create(const Gpu &g, VkDeviceSize bytes) {
        device = g.device;
        size = bytes;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = bytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vkCreateBuffer(device, &bi, nullptr, &buf) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, buf, &req);
        // device-local and host-visible: phones share one DRAM, so this is the GPU's own memory path
        uint32_t type = UINT32_MAX;
        for (const VkMemoryPropertyFlags want :
             {VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT),
              VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)}) {
            for (uint32_t i = 0; i < g.mem.memoryTypeCount && type == UINT32_MAX; ++i)
                if ((req.memoryTypeBits & (1u << i)) && (g.mem.memoryTypes[i].propertyFlags & want) == want) type = i;
            if (type != UINT32_MAX) break;
        }
        if (type == UINT32_MAX) return false;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        if (vkAllocateMemory(device, &ai, nullptr, &memory) != VK_SUCCESS) return false;
        vkBindBufferMemory(device, buf, memory, 0);
        return vkMapMemory(device, memory, 0, bytes, 0, &map) == VK_SUCCESS;
    }
    ~Buffer() {
        if (map) vkUnmapMemory(device, memory);
        if (buf) vkDestroyBuffer(device, buf, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
    }
};

// One compute pipeline over n storage buffers with a push-constant block of push_bytes.
struct Pipeline {
    VkDevice device = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;

    bool create(const Gpu &g, const uint32_t *code, size_t code_bytes, const std::vector<const Buffer *> &bufs,
                uint32_t push_bytes) {
        device = g.device;
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        si.codeSize = code_bytes;
        si.pCode = code;
        if (vkCreateShaderModule(device, &si, nullptr, &shader) != VK_SUCCESS) return false;
        std::vector<VkDescriptorSetLayoutBinding> b(bufs.size());
        for (size_t i = 0; i < bufs.size(); ++i) {
            b[i].binding = static_cast<uint32_t>(i);
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = static_cast<uint32_t>(b.size());
        li.pBindings = b.data();
        if (vkCreateDescriptorSetLayout(device, &li, nullptr, &dsl) != VK_SUCCESS) return false;
        VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &dsl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pc;
        if (vkCreatePipelineLayout(device, &pli, nullptr, &layout) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = shader;
        ci.stage.pName = "main";
        ci.layout = layout;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe) != VK_SUCCESS) return false;
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<uint32_t>(bufs.size())};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 1;
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(device, &pi, nullptr, &dpool) != VK_SUCCESS) return false;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = dpool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &dsl;
        if (vkAllocateDescriptorSets(device, &ai, &set) != VK_SUCCESS) return false;
        std::vector<VkDescriptorBufferInfo> info(bufs.size());
        std::vector<VkWriteDescriptorSet> w(bufs.size());
        for (size_t i = 0; i < bufs.size(); ++i) {
            info[i] = {bufs[i]->buf, 0, VK_WHOLE_SIZE};
            w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[i].dstSet = set;
            w[i].dstBinding = static_cast<uint32_t>(i);
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &info[i];
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
        return true;
    }
    ~Pipeline() {
        if (dpool) vkDestroyDescriptorPool(device, dpool, nullptr);
        if (pipe) vkDestroyPipeline(device, pipe, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (dsl) vkDestroyDescriptorSetLayout(device, dsl, nullptr);
        if (shader) vkDestroyShaderModule(device, shader, nullptr);
    }
};

// Records `reps` dispatches of the pipeline, runs them, returns the seconds per dispatch (or < 0 on failure).
double run(const Gpu &g, const Pipeline &p, uint32_t groups, const void *push, uint32_t push_bytes, int reps) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    if (vkAllocateCommandBuffers(g.device, &ai, &cb) != VK_SUCCESS) return -1;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &bi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.set, 0, nullptr);
    vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    for (int r = 0; r < reps; ++r) {
        vkCmdDispatch(cb, groups, 1, 1);
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb,
                             0, nullptr, 0, nullptr);
    }
    vkEndCommandBuffer(cb);
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(g.device, &fi, nullptr, &fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    const double t0 = now_seconds();
    double dt = -1;
    if (vkQueueSubmit(g.queue, 1, &si, fence) == VK_SUCCESS &&
        vkWaitForFences(g.device, 1, &fence, VK_TRUE, 10'000'000'000ull) == VK_SUCCESS)
        dt = (now_seconds() - t0) / reps;
    vkDestroyFence(g.device, fence, nullptr);
    vkFreeCommandBuffers(g.device, g.pool, 1, &cb);
    return dt;
}

// An arithmetic peak: grows the loop count until a dispatch takes ~40 ms, then times 5 of them.
double peak(const Gpu &g, const uint32_t *code, size_t bytes, const Buffer &out, uint32_t groups, double ops_per_iter) {
    Pipeline p;
    if (!p.create(g, code, bytes, {&out}, 4)) return -1;
    uint32_t iters = 64;
    double dt = run(g, p, groups, &iters, 4, 1);
    while (dt > 0 && dt < 0.04 && iters < (1u << 20)) {
        iters *= 2;
        dt = run(g, p, groups, &iters, 4, 1);
    }
    if (dt <= 0) return -1;
    // warm-up: about half a second of the same work, so the GPU clock is up when it is timed
    run(g, p, groups, &iters, 4, std::max(1, static_cast<int>(0.5 / dt)));
    dt = run(g, p, groups, &iters, 4, 5);
    return dt > 0 ? static_cast<double>(groups) * 256.0 * iters * ops_per_iter / dt : -1;
}

}  // namespace

std::string gpu() {
    Json j;
    j.begin_object();
    Gpu g;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "geode-bench";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &g.instance) != VK_SUCCESS) {
        j.boolean("available", false);
        j.str("reason", "no Vulkan instance");
        j.end_object();
        return j.take();
    }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g.instance, &n, nullptr);
    if (n == 0) {
        j.boolean("available", false);
        j.str("reason", "no Vulkan device");
        j.end_object();
        return j.take();
    }
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(g.instance, &n, devs.data());
    g.phys = devs[0];

    // what it is
    VkPhysicalDeviceSubgroupProperties sub{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext = &sub;
    vkGetPhysicalDeviceProperties2(g.phys, &props);
    const auto &pr = props.properties;
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(g.phys, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> exts(ne);
    vkEnumerateDeviceExtensionProperties(g.phys, nullptr, &ne, exts.data());
    auto has = [&](const char *name) {
        for (const auto &e : exts)
            if (!std::strcmp(e.extensionName, name)) return true;
        return false;
    };
    const uint32_t api = pr.apiVersion;
    const bool f16i8_ext = has("VK_KHR_shader_float16_int8") || api >= VK_API_VERSION_1_2;
    const bool s8_ext = has("VK_KHR_8bit_storage") || api >= VK_API_VERSION_1_2;
    const bool dot_ext = has("VK_KHR_shader_integer_dot_product") || api >= VK_API_VERSION_1_3;

    // which data types its shaders can use
    VkPhysicalDeviceShaderFloat16Int8Features f16i8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice16BitStorageFeatures s16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDevice8BitStorageFeatures s8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    VkPhysicalDeviceShaderIntegerDotProductFeatures dot{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
    VkPhysicalDeviceFeatures2 feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    void **next = &feats.pNext;
    *next = &s16;
    next = &s16.pNext;
    if (f16i8_ext) {
        *next = &f16i8;
        next = &f16i8.pNext;
    }
    if (s8_ext) {
        *next = &s8;
        next = &s8.pNext;
    }
    if (dot_ext) {
        *next = &dot;
        next = &dot.pNext;
    }
    vkGetPhysicalDeviceFeatures2(g.phys, &feats);

    char ver[32];
    std::snprintf(ver, sizeof ver, "%u.%u.%u", VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api), VK_API_VERSION_PATCH(api));
    j.boolean("available", true);
    j.str("name", pr.deviceName);
    j.str("vulkan", ver);
    j.num("vendor_id", pr.vendorID);
    j.num("driver_version", pr.driverVersion);
    j.num("subgroup_size", sub.subgroupSize);
    j.num("max_workgroup_invocations", pr.limits.maxComputeWorkGroupInvocations);
    j.begin_object("types");
    j.boolean("fp32", true);
    j.boolean("fp16_arithmetic", f16i8_ext && f16i8.shaderFloat16);
    j.boolean("fp64", feats.features.shaderFloat64);
    j.boolean("int16_arithmetic", feats.features.shaderInt16);
    j.boolean("int8_arithmetic", f16i8_ext && f16i8.shaderInt8);
    j.boolean("fp16_storage", s16.storageBuffer16BitAccess);
    j.boolean("int8_storage", s8_ext && s8.storageBuffer8BitAccess);
    j.boolean("int8_dot_product", dot_ext && dot.shaderIntegerDotProduct);
    j.end_object();

    // a device with one compute queue, FP16 arithmetic on where it exists
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nq, qf.data());
    g.family = UINT32_MAX;
    for (uint32_t i = 0; i < nq && g.family == UINT32_MAX; ++i)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) g.family = i;
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = g.family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    VkPhysicalDeviceShaderFloat16Int8Features want16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    const bool fp16 = f16i8_ext && f16i8.shaderFloat16;
    want16.shaderFloat16 = fp16 ? VK_TRUE : VK_FALSE;
    std::vector<const char *> dev_exts;
    if (fp16 && api < VK_API_VERSION_1_2) dev_exts.push_back("VK_KHR_shader_float16_int8");
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = fp16 ? &want16 : nullptr;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qi;
    dci.enabledExtensionCount = static_cast<uint32_t>(dev_exts.size());
    dci.ppEnabledExtensionNames = dev_exts.data();
    if (g.family == UINT32_MAX || vkCreateDevice(g.phys, &dci, nullptr, &g.device) != VK_SUCCESS) {
        j.str("error", "cannot create a compute device");
        j.end_object();
        return j.take();
    }
    vkGetDeviceQueue(g.device, g.family, 0, &g.queue);
    vkGetPhysicalDeviceMemoryProperties(g.phys, &g.mem);
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = g.family;
    vkCreateCommandPool(g.device, &cpi, nullptr, &g.pool);

    // the arithmetic peaks: 8 vec4 FMA chains per invocation = 64 FLOP per loop turn
    const uint32_t groups = 512;
    Buffer out;
    if (!out.create(g, static_cast<VkDeviceSize>(groups) * 256 * 16)) {
        j.str("error", "cannot allocate GPU memory");
        j.end_object();
        return j.take();
    }
    j.num("fp32_gflops", peak(g, kSpv_fp32, sizeof kSpv_fp32, out, groups, 64.0) / 1e9);
    if (fp16) j.num("fp16_gflops", peak(g, kSpv_fp16, sizeof kSpv_fp16, out, groups, 64.0) / 1e9);
    j.num("int32_gops", peak(g, kSpv_int32, sizeof kSpv_int32, out, groups, 64.0) / 1e9);

    // memory: 64 MiB read by 1024 groups
    {
        const VkDeviceSize bytes = 64ull << 20;
        Buffer in, sums;
        if (in.create(g, bytes) && sums.create(g, 1024ull * 256 * 16)) {
            std::memset(in.map, 1, bytes);
            Pipeline p;
            if (p.create(g, kSpv_bandwidth, sizeof kSpv_bandwidth, {&in, &sums}, 4)) {
                const uint32_t n4 = static_cast<uint32_t>(bytes / 16);
                const double d1 = run(g, p, 1024, &n4, 4, 1);
                if (d1 > 0) run(g, p, 1024, &n4, 4, std::max(1, static_cast<int>(0.5 / d1)));   // warm-up
                const double dt = run(g, p, 1024, &n4, 4, 5);
                if (dt > 0) j.num("read_gbs", static_cast<double>(bytes) / dt / 1e9);
            }
        }
    }
    j.end_object();
    return j.take();
}

#else

std::string gpu() { return R"({"available":false,"reason":"no Vulkan in this build"})"; }

#endif

}  // namespace geode::bench
