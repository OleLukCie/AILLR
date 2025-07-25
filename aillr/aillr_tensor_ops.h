#ifndef AILLR_TENSOR_OPS_H
#define AILLR_TENSOR_OPS_H

#include <cstddef>
#include <cstdint>
#include <vector>

// ������������ö�٣���չ��������֧�֣�
typedef enum {
    AILLR_DTYPE_FLOAT32,
    AILLR_DTYPE_INT32,
    AILLR_DTYPE_UINT8,
    AILLR_DTYPE_INT8
} AILLR_Dtype;

// ����·���ڵ�ṹ������einsum�Ż���
typedef struct {
    std::vector<size_t> indices;      // ������������������
    std::vector<char> reduce_dims;    // ��Լ���ά�ȱ�ǩ
    std::string einsum_str;           // ��ǰ������einsum���ʽ
    bool use_blas;                    // �Ƿ�ʹ��BLAS����
} AILLR_ContractionNode;

// �����ṹ���壨��ǿ�ڴ�������ά֧�֣�
typedef struct {
    void* data;           // ����ָ�루֧���ⲿ�ڴ棩
    AILLR_Dtype dtype;    // ��������
    size_t* shape;        // ά����״
    int dims;             // ά������
    size_t elem_count;    // Ԫ������
    bool is_external;     // �Ƿ�Ϊ�ⲿ�ڴ棨trueʱ���ڲ��ͷţ�
} AILLR_Tensor;

// ״̬�붨�壨����߽����
typedef enum {
    AILLR_SUCCESS = 0,
    AILLR_NULL_PTR_ERROR,
    AILLR_SHAPE_MISMATCH_ERROR,
    AILLR_TYPE_MISMATCH_ERROR,
    AILLR_MEM_ALLOC_ERROR,
    AILLR_INVALID_AXIS_ERROR,        // ��������Ч������
    AILLR_MEM_LIMIT_EXCEEDED_ERROR,  // �����������ڴ�����
    AILLR_INVALID_CONTRACTION_ERROR  // ��������Ч�������ʽ
} AILLR_Status;

// �������������٣�֧���ⲿ�ڴ棩
AILLR_Status aillr_tensor_create(AILLR_Tensor* tensor, AILLR_Dtype dtype,
    const size_t* shape, int dims);
AILLR_Status aillr_tensor_create_external(AILLR_Tensor* tensor, AILLR_Dtype dtype,
    const size_t* shape, int dims, void* external_data);
AILLR_Status aillr_tensor_destroy(AILLR_Tensor* tensor);

// ��������ӿڣ���չ��ά֧�֣�
AILLR_Status aillr_tensor_add(const AILLR_Tensor* a, const AILLR_Tensor* b, AILLR_Tensor* output);
AILLR_Status aillr_tensor_conv2d(const AILLR_Tensor* input, const AILLR_Tensor* kernel,
    const int* strides, const char* padding, AILLR_Tensor* output);

// �������Խ�����ȡ����ά֧�֣�
AILLR_Status aillr_tensor_diagonal(const AILLR_Tensor* input, int axis1, int axis2, AILLR_Tensor* output);

// ������ת�ã�����֤��
AILLR_Status aillr_tensor_transpose(const AILLR_Tensor* input, const int* axes, AILLR_Tensor* output);

// ������einsum��أ����ļ�����·���Ż���
AILLR_Status aillr_einsum_path(const char* subscripts, const std::vector<AILLR_Tensor*>& operands,
    std::vector<AILLR_ContractionNode>& path, size_t memory_limit);
AILLR_Status aillr_einsum(const char* subscripts, const std::vector<AILLR_Tensor*>& operands,
    AILLR_Tensor* output, bool optimize = false, size_t memory_limit = 1ULL << 30);

#endif // AILLR_TENSOR_OPS_H