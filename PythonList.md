| Reference Order | Library Name | Core Reference Content (Python Layer)                                                                                   | Corresponding AILLR Core Logic                                             |
| --------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 1               | NumPy        | Creation, shape adjustment, and basic operations (addition, subtraction, multiplication, slicing) of tensors (ndarray)  | Core layer `AI_Tensor` structure and basic operations                      |
| 2               | PyTorch      | `torch.Tensor` operations, core operators in `nn.functional` (convolution, matrix multiplication, activation functions) | Implementation logic of basic AI operators (e.g., `ai_operator_conv2d`)    |
| 3               | TensorFlow   | `tf.Tensor` interface, operators in `tf.nn` module (pooling, normalization)                                             | Supplement operator diversity and verify logic consistency                 |
| 4               | ONNX         | Model structure parsing (`onnx.load`), standardized definition of operators                                             | Core layer `ai_model_load` interface and model parsing logic               |
| 5               | TFLite       | Lightweight tensor operations, simplified operator implementations (adapted to mobile scenarios)                        | Reference for "minimalist" interface design of the core layer              |
| 6               | MNN          | Cross-platform model loading, efficient operator (CPU-side) implementation logic                                        | Reference for streamlined implementation of core logic in mobile scenarios |





### 1. Tensor Operations and Basic Numerical Computing: NumPy

NumPy is a fundamental library for tensor (multi-dimensional array) operations in Python, containing core logic such as tensor creation, shape adjustment, and element-wise operations. 

- **Source Code Repository**: [numpy/numpy (GitHub)](https://github.com/numpy/numpy) 
- **Core Logic Locations**: 
- Basic tensor structure definition: `numpy/core/ndarray.py` (Python layer interface) 
- Core operation implementations (C extensions): `numpy/core/src/array/` (e.g., shape adjustment, data access) 
- Mathematical operators (e.g., matrix multiplication, addition, subtraction): `numpy/core/src/umath/` 
  
  ### 2. Core Deep Learning Operators (Convolution, Activation Functions, etc.): PyTorch/TensorFlow
  
  The Python source code of deep learning frameworks includes AI-specific basic operators (convolution, pooling, attention mechanisms, etc.) and model loading logic, which are core references. 
  
  #### PyTorch
- **Source Code Repository**: [pytorch/pytorch (GitHub)](https://github.com/pytorch/pytorch) 
- **Core Logic Locations**: 
- Basic tensor operations: `torch/tensor.py` (Python layer interface), `torch/csrc/api/include/torch/tensor.h` (C++ core definition) 
- Basic operators (e.g., convolution): `torch/nn/functional.py` (Python interface), `torch/csrc/ops/` (C++ implementation) 
- Model loading (e.g., `.pth` file parsing): `torch/serialization.py` 
  
  #### TensorFlow
- **Source Code Repository**: [tensorflow/tensorflow (GitHub)](https://github.com/tensorflow/tensorflow) 
- **Core Logic Locations**: 
- Tensor operations: `tensorflow/python/ops/array_ops.py` 
- Operator implementations (e.g., convolution): `tensorflow/python/ops/nn_ops.py` 
- Model loading (SavedModel/Checkpoint): `tensorflow/python/saved_model/` 
  
  ### 3. Model Format and Cross-Framework Compatibility: ONNX
  
  ONNX (Open Neural Network Exchange) defines a universal model format, and its source code includes model parsing and operator standardization logic, making it suitable for referencing the core logic of "model loading". 
- **Source Code Repository**: [onnx/onnx (GitHub)](https://github.com/onnx/onnx) 
- **Core Logic Locations**: 
- Model structure definition (protobuf): `onnx/onnx.proto` 
- Model loading and validation: `onnx/checker.py`, `onnx/parser.py` 
  
  ### 4. Lightweight Inference Frameworks (for Mobile Scenarios): TFLite/MNN
  
  For mobile AI scenarios, the source code of lightweight inference frameworks is more in line with the needs of "minimal core logic" (e.g., simplified operator implementations, efficient model loading). 
  
  #### TFLite
- **Repository**: [tensorflow/tensorflow (GitHub, TFLite module)](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite) 
- **Core Location**: `tensorflow/lite/kernels/` (basic operator implementations, e.g., `conv.cc`) 
  
  #### MNN
- **Repository**: [alibaba/MNN (GitHub)](https://github.com/alibaba/MNN) 
- **Core Location**: `MNN/src/backend/cpu/` (CPU-side operator implementations), `MNN/tools/converter/` (model conversion/loading logic) 
  
  ### Location Recommendations
1. **Start with "Python layer logic" first**: The Python source code of the above libraries (e.g., `*.py` files) usually reflects functional interfaces and core processes more intuitively (such as the interface definition of PyTorch's `nn.functional.conv2d`), making them suitable as "functional prototypes" for AILLR core logic. 
2. **Ignore hardware optimization details**: Focus on code related to "logical correctness" (e.g., the mathematical calculation process of operators, tensor dimension verification) rather than hardware acceleration (e.g., CUDA kernel functions, assembly optimization), which is consistent with the "hardware-agnostic" design principle of the AILLR core layer. 
3. **Refer to official documentation**: For example, PyTorch's [operator documentation](https://pytorch.org/docs/stable/nn.functional.html) and NumPy's [API manual](https://numpy.org/doc/stable/reference/) can quickly map to core functions in the source code. 
   Through these repositories, you can obtain standard implementations of AI core logic, providing a clear functional reference for the refactoring of AILLR from Python prototypes to the C++ core layer.
