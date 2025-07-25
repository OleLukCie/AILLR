#include "aillr_tensor_ops.h"
#include <cstdlib>
#include <cstring>
#include <map>
#include <algorithm>
#include <stdexcept>

// ��������������Ԫ������
static size_t calculate_elem_count(const size_t* shape, int dims) {
    size_t count = 1;
    for (int i = 0; i < dims; ++i) {
        count *= shape[i];
    }
    return count;
}

// ������������ȡ����Ԫ���ֽڴ�С
static size_t get_dtype_size(AILLR_Dtype dtype) {
    switch (dtype) {
    case AILLR_DTYPE_FLOAT32: return sizeof(float);
    case AILLR_DTYPE_INT32:   return sizeof(int32_t);
    case AILLR_DTYPE_UINT8:   return sizeof(uint8_t);
    case AILLR_DTYPE_INT8:    return sizeof(int8_t);
    default: return 0;
    }
}

// ������������֤��������Ч��
static bool are_axes_valid(const int* axes, int num_axes, int max_dim) {
    if (!axes) return false;
    for (int i = 0; i < num_axes; ++i) {
        if (axes[i] < 0 || axes[i] >= max_dim) return false;
    }
    // ����ظ���
    std::vector<int> unique_axes(axes, axes + num_axes);
    std::sort(unique_axes.begin(), unique_axes.end());
    for (int i = 1; i < num_axes; ++i) {
        if (unique_axes[i] == unique_axes[i - 1]) return false;
    }
    return true;
}

// �����������ڲ��ڴ棩
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

// �����������ⲿ�ڴ棩
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

// ��������
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

// �����ӷ�����������֧�֣�
AILLR_Status aillr_tensor_add(const AILLR_Tensor* a, const AILLR_Tensor* b, AILLR_Tensor* output) {
    // ����У��
    if (!a || !b || !output) return AILLR_NULL_PTR_ERROR;
    if (a->dtype != b->dtype || a->dtype != output->dtype) return AILLR_TYPE_MISMATCH_ERROR;
    if (a->dims != b->dims || a->dims != output->dims) return AILLR_SHAPE_MISMATCH_ERROR;
    for (int i = 0; i < a->dims; ++i) {
        if (a->shape[i] != b->shape[i] || a->shape[i] != output->shape[i]) {
            return AILLR_SHAPE_MISMATCH_ERROR;
        }
    }

    // ������ִ�мӷ�
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

// ��ά�����Խ�����ȡ����3D������ȡi=j��Ԫ�أ�����k�ᣩ
AILLR_Status aillr_tensor_diagonal(const AILLR_Tensor* input, int axis1, int axis2, AILLR_Tensor* output) {
    if (!input || !output) return AILLR_NULL_PTR_ERROR;
    if (axis1 < 0 || axis1 >= input->dims || axis2 < 0 || axis2 >= input->dims) {
        return AILLR_INVALID_AXIS_ERROR;
    }
    if (axis1 == axis2) return AILLR_INVALID_AXIS_ERROR;

    // ȷ�������״���Ƴ�axis2����������ά�ȣ�axis1ά��ȡmin(shape[axis1], shape[axis2])
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

    // �����������
    AILLR_Status status = aillr_tensor_create(output, input->dtype,
        out_shape.data(), out_shape.size());
    if (status != AILLR_SUCCESS) return status;

    // ����Խ���Ԫ�أ���ʵ�֣�ʵ��������ڴ沼���Ż���
    size_t elem_size = get_dtype_size(input->dtype);
    size_t diag_size = out_shape[axis1 < axis2 ? axis1 : axis1 - 1]; // ����axis1������е�λ��
    // ʵ����Ŀ������� strides �����ڴ�ƫ�ƣ��˴���Ϊ˳�����
    for (size_t i = 0; i < diag_size; ++i) {
        // ���������жԽ���Ԫ�ص�������α���룬�����ά��չ����
        size_t input_idx = i * (input->shape[axis2] + 1); // ��ʾ���������ƶ�ά��������
        size_t output_idx = i;
        memcpy((char*)output->data + output_idx * elem_size,
            (char*)input->data + input_idx * elem_size,
            elem_size);
    }
    return AILLR_SUCCESS;
}

// ת�ã�������֤��
AILLR_Status aillr_tensor_transpose(const AILLR_Tensor* input, const int* axes, AILLR_Tensor* output) {
    if (!input || !axes || !output) return AILLR_NULL_PTR_ERROR;
    if (!are_axes_valid(axes, input->dims, input->dims)) {
        return AILLR_INVALID_AXIS_ERROR;
    }

    // ���������״
    std::vector<size_t> out_shape(input->dims);
    for (int i = 0; i < input->dims; ++i) {
        out_shape[i] = input->shape[axes[i]];
    }

    // �����������
    AILLR_Status status = aillr_tensor_create(output, input->dtype,
        out_shape.data(), input->dims);
    if (status != AILLR_SUCCESS) return status;

    // ִ��ת�ã�ʵ��������ڴ沼���Ż����˴��򻯣�
    // ע����ʵʵ������������������ӳ�䣬����������/������
    size_t elem_size = get_dtype_size(input->dtype);
    memcpy(output->data, input->data, input->elem_count * elem_size); // ��ʾ����������
    return AILLR_SUCCESS;
}

// ̰���㷨ѡ�������ԣ�����ʵ��ά�ȼ���ɱ���
static size_t calculate_contraction_cost(const std::vector<AILLR_Tensor*>& tensors,
    const std::vector<char>& shared_dims,
    const std::map<char, size_t>& dim_map) {
    size_t cost = 1;
    // �����ɱ� = ����ά�ȴ�С�˻� * ʣ��ά�ȴ�С�˻�
    for (char d : shared_dims) {
        cost *= dim_map.at(d); // ����ά�ȳ˻�
    }
    for (const auto& tensor : tensors) {
        for (int i = 0; i < tensor->dims; ++i) {
            // ����dim_map��key��ά�ȱ�ǩ����ӳ�䵽ʵ��ά��������ʾ���߼���
            cost *= tensor->shape[i];
        }
    }
    return cost;
}

// einsum·�����㣨�����ڴ����ƣ�
AILLR_Status aillr_einsum_path(const char* subscripts, const std::vector<AILLR_Tensor*>& operands,
    std::vector<AILLR_ContractionNode>& path, size_t memory_limit) {
    // 1. ����subscripts��ȡά�ȱ�ǩ������ӳ�䣨ʡ�Ծ���ʵ�֣�
    // 2. ����ά�ȴ�Сӳ���dim_map: ��ǩ -> �ߴ磩
    std::map<char, size_t> dim_map;
    // 3. ̰��ѡ��������
    while (operands.size() > 1) {
        // �ҵ��ɱ���͵������ԣ�ʾ���߼���
        size_t min_cost = SIZE_MAX;
        AILLR_ContractionNode best_node;
        for (size_t i = 0; i < operands.size(); ++i) {
            for (size_t j = i + 1; j < operands.size(); ++j) {
                // ���㹲��ά��
                std::vector<char> shared_dims; // ʵ�����subscripts��ȡ
                size_t cost = calculate_contraction_cost({ operands[i], operands[j] },
                    shared_dims, dim_map);
                // ����ڴ�����
                size_t intermediate_size = cost * get_dtype_size(operands[i]->dtype);
                if (intermediate_size > memory_limit) {
                    continue; // ���������ڴ����Ƶ�������
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
        // ģ���������������������ʵ�������operands�б�
    }
    return AILLR_SUCCESS;
}

// einsum���ļ��㣨��������·����
AILLR_Status aillr_einsum(const char* subscripts, const std::vector<AILLR_Tensor*>& operands,
    AILLR_Tensor* output, bool optimize, size_t memory_limit) {
    if (!subscripts || operands.empty() || !output) return AILLR_NULL_PTR_ERROR;

    // 1. ��������·��
    std::vector<AILLR_ContractionNode> path;
    AILLR_Status status = aillr_einsum_path(subscripts, operands, path, memory_limit);
    if (status != AILLR_SUCCESS) return status;

    // 2. ִ����������
    std::vector<AILLR_Tensor*> tmp_operands = operands; // ��ʱ�����б�
    for (const auto& node : path) {
        // ��ȡ��������������
        std::vector<AILLR_Tensor*> current_tensors;
        for (size_t idx : node.indices) {
            current_tensors.push_back(tmp_operands[idx]);
        }

        // ִ������������˷�/ά��Լ��ȣ�
        AILLR_Tensor intermediate;
        if (node.use_blas) {
            // ʹ��BLAS���پ���˷���ʾ����
            status = aillr_tensor_matmul(current_tensors[0], current_tensors[1], &intermediate);
        }
        else {
            // ͨ�������߼�������einsum_str��
            status = aillr_tensor_contraction(node.einsum_str.c_str(), current_tensors, &intermediate);
        }
        if (status != AILLR_SUCCESS) return status;

        // ������ʱ�����б�
        // ��ʡ�ԣ��Ƴ�����������������м�����
    }

    // 3. ������ս��
    memcpy(output->data, tmp_operands[0]->data, output->elem_count * get_dtype_size(output->dtype));
    return AILLR_SUCCESS;
}

// ������㣨�������ʵ�֣�
AILLR_Status aillr_tensor_conv2d(const AILLR_Tensor* input, const AILLR_Tensor* kernel,
    const int* strides, const char* padding, AILLR_Tensor* output) {
    // ����У��
    if (!input || !kernel || !strides || !padding || !output) return AILLR_NULL_PTR_ERROR;
    if (input->dims != 4 || kernel->dims != 4) return AILLR_SHAPE_MISMATCH_ERROR; // NCHW��ʽ

    // ���������״����ʵ�֣�
    size_t out_h = (input->shape[2] - kernel->shape[2] + 2 * (strcmp(padding, "SAME") ? 0 : 1)) / strides[0] + 1;
    size_t out_w = (input->shape[3] - kernel->shape[3] + 2 * (strcmp(padding, "SAME") ? 0 : 1)) / strides[1] + 1;
    size_t out_shape[] = { input->shape[0], kernel->shape[0], out_h, out_w };
    AILLR_Status status = aillr_tensor_create(output, input->dtype, out_shape, 4);
    if (status != AILLR_SUCCESS) return status;

    // ������㣨ʡ�Ծ���ʵ�֣�ʵ����չ��ѭ�������Ӳ�����ٽӿڣ�
    return AILLR_SUCCESS;
}