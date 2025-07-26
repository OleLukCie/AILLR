## AI Low-Level Refactoring (**AILLR**)
---
#Experimental code is placed in the "dev" branch https://github.com/OleLukCie/AILLR/tree/dev
#and will not be merged into the main branch until fully implemented.

###### If there are any errors or inadequacies, please point them out.

---

Currently designed rules:

1. Interface Design: **Abstract, Minimal, Stable**

The interfaces (APIs) exposed by the core layer to the outside need to meet three principles:

- **Abstraction**: Interfaces only define "what to do" without restricting "how to do it". They do not specify the specific hardware implementation of manufacturers, leaving flexible choices for the adapter layer.
  
- **Minimization**: Only retain necessary interfaces to avoid over-design. Do not include functions that may be differentiated by manufacturers (these can be implemented through extended interfaces of the adapter layer).
  
- **Stability**: Once interfaces are released, avoid destructive changes as much as possible. If iteration is needed, adopt a "version compatibility" strategy, retaining old interfaces to prevent frequent rework on the manufacturer's adapter layer.
  
  Example interfaces (pseudocode):
  
  ```cpp
  typedef struct {
    int width;
    int height;
    float* data;
  } AI_Tensor;
  
  int ai_operator_conv2d(const AI_Tensor* input, const AI_Tensor* kernel, AI_Tensor* output);
  
  int ai_model_load(AI_Model* model, const char* path);
  ```
  

2. Open-Source Collaboration

- **License**: Apache 2.0
- Propose interface improvement suggestions through GitHub Issues, etc.
- The core layer code is fully open-source for code auditing and technical improvements.

**Abstraction Layer Encapsulation**: Encapsulate underlying C++ logic into APIs, hide complex features, and allow developers to use core functions by calling interfaces.

**Layered Support System**: Provide differentiated documents for different roles (application developers refer to API manuals, secondary developers refer to interface principles, core maintainers refer to C++ design specifications), and establish a "newbie-friendly" community mutual assistance mechanism (such as PR guidance, common error checklists).

The core functions of any programming language are essentially implementations of "computational logic", and computational logic itself is language-agnostic. Therefore, **as long as the functional logic of a Python library is clear, it can theoretically be re-implemented in C++**, including declaring interfaces through C++ header files (`.h`) and implementing functions in source files (`.cpp`).
