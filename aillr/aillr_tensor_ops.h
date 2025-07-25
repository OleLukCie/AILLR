#ifndef AILLR_TENSOR_OPS_H
#define AILLR_TENSOR_OPS_H

#include <cstddef>
#include <cstdint>
#include <vector>

// 张量数据类型枚举（扩展基础类型支持）
typedef enum {
    AILLR_DTYPE_FLOAT32,
    AILLR_DTYPE_INT32,
    AILLR_DTYPE_UINT8,
    AILLR_DTYPE_INT8
} AILLR_Dtype;

// 收缩路径节点结构（用于einsum优化）
typedef struct {
    std::vector<size_t> indices;      // 参与收缩的张量索引
    std::vector<char> reduce_dims;    // 待约简的维度标签
    std::string einsum_str;           // 当前收缩的einsum表达式
    bool use_blas;                    // 是否使用BLAS加速
} AILLR_ContractionNode;

// 张量结构定义（增强内存管理与多维支持）
typedef struct {
    void* data;           // 数据指针（支持外部内存）
    AILLR_Dtype dtype;    // 数据类型
    size_t* shape;        // 维度形状
    int dims;             // 维度数量
    size_t elem_count;    // 元素总数
    bool is_external;     // 是否为外部内存（true时不内部释放）
} AILLR_Tensor;

// 状态码定义（补充边界错误）
typedef enum {
    AILLR_SUCCESS = 0,
    AILLR_NULL_PTR_ERROR,
    AILLR_SHAPE_MISMATCH_ERROR,
    AILLR_TYPE_MISMATCH_ERROR,
    AILLR_MEM_ALLOC_ERROR,
    AILLR_INVALID_AXIS_ERROR,        // 新增：无效轴索引
    AILLR_MEM_LIMIT_EXCEEDED_ERROR,  // 新增：超过内存限制
    AILLR_INVALID_CONTRACTION_ERROR  // 新增：无效收缩表达式
} AILLR_Status;

// 张量创建与销毁（支持外部内存）
AILLR_Status aillr_tensor_create(AILLR_Tensor* tensor, AILLR_Dtype dtype,
    const size_t* shape, int dims);
AILLR_Status aillr_tensor_create_external(AILLR_Tensor* tensor, AILLR_Dtype dtype,
    const size_t* shape, int dims, void* external_data);
AILLR_Status aillr_tensor_destroy(AILLR_Tensor* tensor);

// 核心运算接口（扩展多维支持）
AILLR_Status aillr_tensor_add(const AILLR_Tensor* a, const AILLR_Tensor* b, AILLR_Tensor* output);
AILLR_Status aillr_tensor_conv2d(const AILLR_Tensor* input, const AILLR_Tensor* kernel,
    const int* strides, const char* padding, AILLR_Tensor* output);

// 新增：对角线提取（多维支持）
AILLR_Status aillr_tensor_diagonal(const AILLR_Tensor* input, int axis1, int axis2, AILLR_Tensor* output);

// 新增：转置（轴验证）
AILLR_Status aillr_tensor_transpose(const AILLR_Tensor* input, const int* axes, AILLR_Tensor* output);

// 新增：einsum相关（核心计算与路径优化）
AILLR_Status aillr_einsum_path(const char* subscripts, const std::vector<AILLR_Tensor*>& operands,
    std::vector<AILLR_ContractionNode>& path, size_t memory_limit);
AILLR_Status aillr_einsum(const char* subscripts, const std::vector<AILLR_Tensor*>& operands,
    AILLR_Tensor* output, bool optimize = false, size_t memory_limit = 1ULL << 30);

#endif // AILLR_TENSOR_OPS_H