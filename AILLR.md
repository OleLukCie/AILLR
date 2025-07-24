## AI Low-Level Refactoring (**AILLR**)

Definition: For AI scenarios, it refers to the re-implementation of core logic from high-level user-friendly languages (Python) to low-level high-performance languages (C++).

- "AI": the field of artificial intelligence;
- "Low-level": the low-level features of the target language (C++), such as being hardware-close, high-performance, and system-level;
- "Refactoring": re-implementing in a way more suitable for the scenario while keeping the functional logic unchanged.

Core logic: First explore and implement the core logic of AI functions in Python, then re-implement the core logic in C++ while maintaining functional consistency.

Purpose: To decentralize the control of AI technology from a few giants to the entire community and manufacturers. Mobile phones are the most inclusive intelligent hardware; implementing the design on mobile phones allows "democratic AI" to move from an idea to the public (mobile phone + cloud = smart phone).

**Dual-Layer Architecture Design**:

- Core Layer: Implement general logic in standard C++, with strictly standardized code to ensure long-term stability.
- Adapter Layer: Interface implementation for specific hardware.

**Core maintainers focus on the core layer to ensure standardization and stability**

The core layer implements general logic in standard C++ and does not involve any hardware/system-related code:

- All manufacturers develop based on the same set of core logic.
- As core maintainers, they can focus on logic optimization, bug fixes, and long-term iterations, unaffected by specific hardware changes, ensuring long-term stability.
- Concentrate efforts on maintaining core logic to avoid redundant development of basic functions by manufacturers.

**Manufacturers are responsible for the adapter layer**

Manufacturers (such as mobile phone brands, chip manufacturers, embedded device vendors) write adaptation code for their own hardware (e.g., NPU driver calls, sensor data parsing, hardware acceleration logic) based on the standard interfaces of the core layer. Because manufacturers know their own hardware characteristics best, and different manufacturers have different product positioning.

<u>To make this model run smoothly, the core is to design good interfaces between the core layer and the adapter layer</u>

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



------------------------------------------------------------------------------------------------------------------------------

###### If there are any errors or inadequacies, please point them out.

------------------------------------------------------------------------------------------------------------------------------

Dēorlic gemynd, of nihte cōm

John

July 25, 2025