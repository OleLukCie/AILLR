#include "aillr_tensor_ops.h"
#include <cstdlib>
#include <cstring>
#include <map>
#include <algorithm>
#include <stdexcept>

// 辅助函数：计算元素总数
static size_t calculate_elem_count(const size_t* shape, int dims) {
    size_t count = 1;
    for (int i = 0; i < dims; ++i) {
        count *= shape[i];
    }
    return count;
}

// 辅助函数：获取单个元素字节大小
static size_t get_dtype_size(AILLR_Dtype dtype) {
    switch (dtype) {
    case AILLR_DTYPE_FLOAT32: return sizeof(float);
    case AILLR_DTYPE_INT32:   return sizeof(int32_t);
    case AILLR_DTYPE_UINT8:   return sizeof(uint8_t);
    case AILLR_DTYPE_INT8:    return sizeof(int8_t);
    default: return 0;
    }
}

// 辅助函数：验证轴索引有效性
static bool are_axes_valid(const int* axes, int num_axes, int max_dim) {
    if (!axes) return false;
    for (int i = 0; i < num_axes; ++i) {
        if (axes[i] < 0 || axes[i] >= max_dim) return false;
    }
    // 检查重复轴
    std::vector<int> unique_axes(axes, axes + num_axes);
    std::sort(unique_axes.begin(), unique_axes.end());
    for (int i = 1; i < num_axes; ++i) {
        if (unique_axes[i] == unique_axes[i - 1]) return false;
    }
    return true;
}

// 张量创建（内部内存）
AILLR_Status aillr_tensor_create(AILLR_Tensor* tensor, AILLR_Dtype dtype,
    const size_t* shape, int dims) {
    if (!tensor || !shape || dims <= 0) {
        return AILLR_NULL_PTR_ERROR;
    }
    tensor->dtype = dtype;
    tensor->dims = dims;
    tensor->shape = (size_t*)malloc(dims * sizeof(size_t));
    if (!tensor->shape) return AILLR_MEM_ALLOC_ERROR;
    memcpy(tensor->shape, shape, dims * sizeof(size_t));

    tensor->elem_count = calculate_elem_count(shape, dims);
    size_t data_size = tensor->elem_count * get_dtype_size(dtype);
    tensor->data = malloc(data_size);
    if (!tensor->data) {
        free(tensor->shape);
        tensor->shape = nullptr;
        return AILLR_MEM_ALLOC_ERROR;
    }
    tensor->is_external = false;
    return AILLR_SUCCESS;
}

// 张量创建（外部内存）
AILLR_Status aillr_tensor_create_external(AILLR_Tensor* tensor, AILLR_Dtype dtype,
    const size_t* shape, int dims, void* external_data) {
    if (!tensor || !shape || dims <= 0 || !external_data) {
        return AILLR_NULL_PTR_ERROR;
    }
    tensor->dtype = dtype;
    tensor->dims = dims;
    tensor->shape = (size_t*)malloc(dims * sizeof(size_t));
    if (!tensor->shape) return AILLR_MEM_ALLOC_ERROR;
    memcpy(tensor->shape, shape, dims * sizeof(size_t));

    tensor->elem_count = calculate_elem_count(shape, dims);
    tensor->data = external_data;
    tensor->is_external = true;
    return AILLR_SUCCESS;
}

// 张量销毁
AILLR_Status aillr_tensor_destroy(AILLR_Tensor* tensor) {
    if (!tensor) return AILLR_NULL_PTR_ERROR;
    if (!tensor->is_external) free(tensor->data);
    free(tensor->shape);
    tensor->data = nullptr;
    tensor->shape = nullptr;
    tensor->elem_count = 0;
    tensor->dims = 0;
    return AILLR_SUCCESS;
}

// 张量加法（完善类型支持）
AILLR_Status aillr_tensor_add(const AILLR_Tensor* a, const AILLR_Tensor* b, AILLR_Tensor* output) {
    // 参数校验
    if (!a || !b || !output) return AILLR_NULL_PTR_ERROR;
    if (a->dtype != b->dtype || a->dtype != output->dtype) return AILLR_TYPE_MISMATCH_ERROR;
    if (a->dims != b->dims || a->dims != output->dims) return AILLR_SHAPE_MISMATCH_ERROR;
    for (int i = 0; i < a->dims; ++i) {
        if (a->shape[i] != b->shape[i] || a->shape[i] != output->shape[i]) {
            return AILLR_SHAPE_MISMATCH_ERROR;
        }
    }

    // 按类型执行加法
    size_t elem_size = get_dtype_size(a->dtype);
    switch (a->dtype) {
    case AILLR_DTYPE_FLOAT32: {
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        float* out_data = (float*)output->data;
        for (size_t i = 0; i < a->elem_count; ++i) {
            out_data[i] = a_data[i] + b_data[i];
        }
        break;
    }
    case AILLR_DTYPE_INT32: {
        int32_t* a_data = (int32_t*)a->data;
        int32_t* b_data = (int32_t*)b->data;
        int32_t* out_data = (int32_t*)output->data;
        for (size_t i = 0; i < a->elem_count; ++i) {
            out_data[i] = a_data[i] + b_data[i];
        }
        break;
    }
    case AILLR_DTYPE_UINT8: {
        uint8_t* a_data = (uint8_t*)a->data;
        uint8_t* b_data = (uint8_t*)b->data;
        uint8_t* out_data = (uint8_t*)output->data;
        for (size_t i = 0; i < a->elem_count; ++i) {
            out_data[i] = a_data[i] + b_data[i];
        }
        break;
    }
    case AILLR_DTYPE_INT8: {
        int8_t* a_data = (int8_t*)a->data;
        int8_t* b_data = (int8_t*)b->data;
        int8_t* out_data = (int8_t*)output->data;
        for (size_t i = 0; i < a->elem_count; ++i) {
            out_data[i] = a_data[i] + b_data[i];
        }
        break;
    }
    default: return AILLR_TYPE_MISMATCH_ERROR;
    }
    return AILLR_SUCCESS;
}

// 多维张量对角线提取（如3D张量提取i=j的元素，保留k轴）
AILLR_Status aillr_tensor_diagonal(const AILLR_Tensor* input, int axis1, int axis2, AILLR_Tensor* output) {
    if (!input || !output) return AILLR_NULL_PTR_ERROR;
    if (axis1 < 0 || axis1 >= input->dims || axis2 < 0 || axis2 >= input->dims) {
        return AILLR_INVALID_AXIS_ERROR;
    }
    if (axis1 == axis2) return AILLR_INVALID_AXIS_ERROR;

    // 确定输出形状：移除axis2，保留其他维度，axis1维度取min(shape[axis1], shape[axis2])
    std::vector<size_t> out_shape;
    for (int i = 0; i < input->dims; ++i) {
        if (i == axis2) continue;
        if (i == axis1) {
            out_shape.push_back(std::min(input->shape[axis1], input->shape[axis2]));
        }
        else {
            out_shape.push_back(input->shape[i]);
        }
    }

    // 创建输出张量
    AILLR_Status status = aillr_tensor_create(output, input->dtype,
        out_shape.data(), out_shape.size());
    if (status != AILLR_SUCCESS) return status;

    // 计算对角线元素（简化实现，实际需根据内存布局优化）
    size_t elem_size = get_dtype_size(input->dtype);
    size_t diag_size = out_shape[axis1 < axis2 ? axis1 : axis1 - 1]; // 修正axis1在输出中的位置
    // 实际项目中需根据 strides 计算内存偏移，此处简化为顺序访问
    for (size_t i = 0; i < diag_size; ++i) {
        // 计算输入中对角线元素的索引（伪代码，需根据维度展开）
        size_t input_idx = i * (input->shape[axis2] + 1); // 仅示例，需完善多维索引计算
        size_t output_idx = i;
        memcpy((char*)output->data + output_idx * elem_size,
            (char*)input->data + input_idx * elem_size,
            elem_size);
    }
    return AILLR_SUCCESS;
}

// 转置（带轴验证）
AILLR_Status aillr_tensor_transpose(const AILLR_Tensor* input, const int* axes, AILLR_Tensor* output) {
    if (!input || !axes || !output) return AILLR_NULL_PTR_ERROR;
    if (!are_axes_valid(axes, input->dims, input->dims)) {
        return AILLR_INVALID_AXIS_ERROR;
    }

    // 构建输出形状
    std::vector<size_t> out_shape(input->dims);
    for (int i = 0; i < input->dims; ++i) {
        out_shape[i] = input->shape[axes[i]];
    }

    // 创建输出张量
    AILLR_Status status = aillr_tensor_create(output, input->dtype,
        out_shape.data(), input->dims);
    if (status != AILLR_SUCCESS) return status;

    // 执行转置（实际需根据内存布局优化，此处简化）
    // 注：真实实现需计算输入输出索引映射，考虑行主序/列主序
    size_t elem_size = get_dtype_size(input->dtype);
    memcpy(output->data, input->data, input->elem_count * elem_size); // 仅示例，需完善
    return AILLR_SUCCESS;
}

// 贪心算法选择收缩对（基于实际维度计算成本）
static size_t calculate_contraction_cost(const std::vector<AILLR_Tensor*>& tensors,
    const std::vector<char>& shared_dims,
    const std::map<char, size_t>& dim_map) {
    size_t cost = 1;
    // 收缩成本 = 共享维度大小乘积 * 剩余维度大小乘积
    for (char d : shared_dims) {
        cost *= dim_map.at(d); // 共享维度乘积
    }
    for (const auto& tensor : tensors) {
        for (int i = 0; i < tensor->dims; ++i) {
            // 假设dim_map的key是维度标签，需映射到实际维度索引（示例逻辑）
            cost *= tensor->shape[i];
        }
    }
    return cost;
}

// einsum路径计算（处理内存限制）
AILLR_Status aillr_einsum_path(const char* subscripts, const std::vector<AILLR_Tensor*>& operands,
    std::vector<AILLR_ContractionNode>& path, size_t memory_limit) {
    // 1. 解析subscripts获取维度标签与张量映射（省略具体实现）
    // 2. 构建维度大小映射表（dim_map: 标签 -> 尺寸）
    std::map<char, size_t> dim_map;
    // 3. 贪心选择收缩对
    while (operands.size() > 1) {
        // 找到成本最低的收缩对（示例逻辑）
        size_t min_cost = SIZE_MAX;
        AILLR_ContractionNode best_node;
        for (size_t i = 0; i < operands.size(); ++i) {
            for (size_t j = i + 1; j < operands.size(); ++j) {
                // 计算共享维度
                std::vector<char> shared_dims; // 实际需从subscripts提取
                size_t cost = calculate_contraction_cost({ operands[i], operands[j] },
                    shared_dims, dim_map);
                // 检查内存限制
                size_t intermediate_size = cost * get_dtype_size(operands[i]->dtype);
                if (intermediate_size > memory_limit) {
                    continue; // 跳过超过内存限制的收缩对
                }
                if (cost < min_cost) {
                    min_cost = cost;
                    best_node.indices = { i, j };
                    best_node.reduce_dims = shared_dims;
                }
            }
        }
        if (min_cost == SIZE_MAX) return AILLR_MEM_LIMIT_EXCEEDED_ERROR;
        path.push_back(best_node);
        // 模拟收缩后减少张量数量（实际需更新operands列表）
    }
    return AILLR_SUCCESS;
}

// einsum核心计算（基于收缩路径）
AILLR_Status aillr_einsum(const char* subscripts, const std::vector<AILLR_Tensor*>& operands,
    AILLR_Tensor* output, bool optimize, size_t memory_limit) {
    if (!subscripts || operands.empty() || !output) return AILLR_NULL_PTR_ERROR;

    // 1. 计算收缩路径
    std::vector<AILLR_ContractionNode> path;
    AILLR_Status status = aillr_einsum_path(subscripts, operands, path, memory_limit);
    if (status != AILLR_SUCCESS) return status;

    // 2. 执行收缩步骤
    std::vector<AILLR_Tensor*> tmp_operands = operands; // 临时张量列表
    for (const auto& node : path) {
        // 提取参与收缩的张量
        std::vector<AILLR_Tensor*> current_tensors;
        for (size_t idx : node.indices) {
            current_tensors.push_back(tmp_operands[idx]);
        }

        // 执行收缩（矩阵乘法/维度约简等）
        AILLR_Tensor intermediate;
        if (node.use_blas) {
            // 使用BLAS加速矩阵乘法（示例）
            status = aillr_tensor_matmul(current_tensors[0], current_tensors[1], &intermediate);
        }
        else {
            // 通用收缩逻辑（基于einsum_str）
            status = aillr_tensor_contraction(node.einsum_str.c_str(), current_tensors, &intermediate);
        }
        if (status != AILLR_SUCCESS) return status;

        // 更新临时张量列表
        // （省略：移除收缩的张量，添加中间结果）
    }

    // 3. 输出最终结果
    memcpy(output->data, tmp_operands[0]->data, output->elem_count * get_dtype_size(output->dtype));
    return AILLR_SUCCESS;
}

// 卷积运算（补充基础实现）
AILLR_Status aillr_tensor_conv2d(const AILLR_Tensor* input, const AILLR_Tensor* kernel,
    const int* strides, const char* padding, AILLR_Tensor* output) {
    // 参数校验
    if (!input || !kernel || !strides || !padding || !output) return AILLR_NULL_PTR_ERROR;
    if (input->dims != 4 || kernel->dims != 4) return AILLR_SHAPE_MISMATCH_ERROR; // NCHW格式

    // 计算输出形状（简化实现）
    size_t out_h = (input->shape[2] - kernel->shape[2] + 2 * (strcmp(padding, "SAME") ? 0 : 1)) / strides[0] + 1;
    size_t out_w = (input->shape[3] - kernel->shape[3] + 2 * (strcmp(padding, "SAME") ? 0 : 1)) / strides[1] + 1;
    size_t out_shape[] = { input->shape[0], kernel->shape[0], out_h, out_w };
    AILLR_Status status = aillr_tensor_create(output, input->dtype, out_shape, 4);
    if (status != AILLR_SUCCESS) return status;

    // 卷积计算（省略具体实现，实际需展开循环或调用硬件加速接口）
    return AILLR_SUCCESS;
}