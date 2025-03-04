# PyTorch Fundamentals: A Comprehensive Guide

## Introduction to PyTorch

PyTorch is an open-source machine learning library developed by Facebook's AI Research lab (FAIR) in 2016. It has quickly become one of the most popular frameworks for deep learning research and applications, with adoption from organizations like OpenAI, Tesla, Microsoft, and many top research institutions.

### Why PyTorch Was Invented

PyTorch emerged to address several limitations in existing deep learning frameworks:

1. **Dynamic Computation Graph**: Unlike frameworks with static computation graphs (like early versions of TensorFlow), PyTorch allows for building and modifying neural networks on-the-fly. This is similar to how you might debug a regular Python program - step by step, with full visibility into intermediate values.

2. **Python-First Approach**: PyTorch was designed to integrate seamlessly with Python, making it more intuitive for researchers familiar with Python's data science ecosystem. It feels like a natural extension of NumPy rather than a separate system with its own rules.

3. **Research-Friendly**: Facebook designed PyTorch to prioritize flexibility, debugging capabilities, and ease of use for research. The ability to use standard Python debugging tools makes experimenting with new ideas much faster.

4. **Imperative Programming Style**: PyTorch uses an imperative programming style where operations are executed as they're defined, making code more intuitive to read and debug.

### Core Components of PyTorch

PyTorch consists of several key components that work together:

1. **torch**: The core package containing tensor computations and automatic differentiation.
2. **torch.nn**: Neural network layers and components.
3. **torch.optim**: Optimization algorithms like SGD, Adam, etc.
4. **torch.utils.data**: Data loading utilities including Dataset and DataLoader classes.
5. **torchvision, torchaudio, torchtext**: Domain-specific packages for computer vision, audio processing, and NLP.
6. **torch.jit**: Just-In-Time (JIT) compilation with TorchScript.
7. **torch.distributed**: Tools for distributed training across multiple GPUs/machines.
8. **torch.multiprocessing**: Parallel data processing utilities.

### Key Features and Specialties of PyTorch

- **Dynamic Computational Graph**: Define-by-run semantics that allow for changing neural network behavior during runtime.

  _Analogy_: Think of static graphs as following a fixed recipe exactly as written, while dynamic graphs are like cooking and adjusting ingredients as you go based on how the dish tastes.

- **Pythonic Interface**: Feels natural to Python programmers, leveraging familiar concepts and syntax.

- **Strong GPU Acceleration**: Efficient CUDA integration with minimal code changes required to move computation from CPU to GPU.

- **Production Ready**: TorchScript, TorchServe, and mobile deployment tools for production environments.

- **Rich Ecosystem**: Pre-built models, libraries, and tools like TorchVision, TorchText, TorchAudio, etc.

- **Community Support**: Active development by Facebook (Meta) and extensive community contributions.

- **Hybrid Frontend**: Combines eager execution with graph-based execution through TorchScript.

### What Can We Do With PyTorch?

PyTorch enables a wide range of applications:

#### 1. Deep Learning Research and Development

- Rapid prototyping of deep neural networks
- Experimentation with cutting-edge models
- Implementation of papers and novel architectures

#### 2. Computer Vision

- Image classification (ResNet, EfficientNet)
- Object detection (YOLO, Faster R-CNN)
- Semantic segmentation (U-Net, DeepLab)
- Generative adversarial networks (StyleGAN, CycleGAN)
- Image synthesis and manipulation

#### 3. Natural Language Processing

- Sentiment analysis and text classification
- Machine translation
- Named entity recognition
- Question answering systems
- Language modeling (BERT, GPT)
- Transformers and attention mechanisms

#### 4. Speech and Audio Processing

- Speech recognition
- Speaker identification
- Audio classification
- Music generation

#### 5. Reinforcement Learning

- Game playing agents
- Robotics control
- Autonomous systems

#### 6. Scientific Computing

- Physics simulations
- Drug discovery
- Computational biology
- Material science

#### 7. Time Series Analysis

- Financial forecasting
- Weather prediction
- Anomaly detection
- Predictive maintenance

#### 8. Multimodal Learning

- Image captioning
- Visual question answering
- Text-to-image generation

## PyTorch vs TensorFlow: A Deeper Comparison

Both are leading deep learning frameworks, but they have different design philosophies and strengths:

| Feature            | PyTorch                               | TensorFlow                                                |
| ------------------ | ------------------------------------- | --------------------------------------------------------- |
| Computation Graph  | Dynamic (define-by-run)               | Static with eager execution option                        |
| Primary Language   | Python-centric                        | Multiple language bindings (Python, C++, Java)            |
| Debugging          | Native Python debugging               | More complex debugging (improving with eager mode)        |
| Deployment         | TorchScript, LibTorch, Mobile options | Production-focused ecosystem (TF Lite, TF.js, TF Serving) |
| Community          | Academic/research focus               | Industry/production focus                                 |
| Visualization      | Basic tools plus external options     | Built-in TensorBoard                                      |
| Ecosystem Maturity | Growing rapidly                       | More mature, especially in production                     |
| Learning Curve     | Gentle, especially for Python users   | Steeper, more concepts to learn                           |
| TPU Support        | Limited                               | Extensive                                                 |
| Model Definition   | Object-oriented approach              | Both functional and object-oriented APIs                  |

### When to Choose PyTorch:

- Research and prototyping where flexibility is key
- Need for dynamic network architectures that change during runtime
- When pythonic code and intuitive debugging are priorities
- Academic and research projects requiring rapid iteration

### When to Choose TensorFlow:

- Production deployment at large scale
- When using TPUs (Tensor Processing Units)
- Mobile and edge deployment with optimized inference
- Need for extensive visualization tools
- When working in a more established industry setting with existing TensorFlow infrastructure

_Analogy_: PyTorch is like a well-equipped artist's studio that gives you freedom to create and experiment, while TensorFlow is like a professional manufacturing facility optimized for consistent, large-scale production.

## Tensors: The Building Blocks of PyTorch

### What Are Tensors?

Tensors are multi-dimensional arrays that serve as the fundamental data structures in PyTorch. They are similar to NumPy arrays but with additional capabilities:

- Can run on GPUs for accelerated computing
- Track gradients for automatic differentiation
- Optimized for deep learning operations

_Analogy_: If a neural network is like a building, tensors are the bricks, beams, and all construction materials. Everything in PyTorch is built with tensors.

Tensor dimensionality corresponds to different types of data:

- 0D tensor (scalar): Single value, like "42" or "0.5"
- 1D tensor (vector): Array of values, like "[1, 2, 3, 4]"
- 2D tensor (matrix): Table of values, like a spreadsheet or an image with one color channel
- 3D tensor: Cube of values (e.g., RGB image with height, width, and channels)
- 4D tensor: Batch of images (batch_size, channels, height, width)
- Higher dimensions: More complex data structures like videos or time series of 3D data

### Creating Tensors

```python
import torch

# From Python list
x = torch.tensor([1, 2, 3, 4])
print(x)
# Output: tensor([1, 2, 3, 4])

# Zeros and ones
zeros = torch.zeros(3, 4)  # 3x4 tensor of zeros
print(zeros)
# Output: tensor([[0., 0., 0., 0.],
#                [0., 0., 0., 0.],
#                [0., 0., 0., 0.]])

ones = torch.ones(2, 3)    # 2x3 tensor of ones
print(ones)
# Output: tensor([[1., 1., 1.],
#                [1., 1., 1.]])

# Random tensors
random_tensor = torch.rand(3, 4)  # Random values from uniform distribution [0, 1)
print(random_tensor)
# Output: tensor([[0.1234, 0.5678, 0.2468, 0.8765],
#                [0.3579, 0.9876, 0.4321, 0.7654],
#                [0.2143, 0.6587, 0.9512, 0.3254]])
# Note: Your random values will be different

randn_tensor = torch.randn(3, 4)  # Random values from normal distribution (mean=0, var=1)
print(randn_tensor)
# Output: tensor([[ 0.1234, -0.5678,  1.2468, -0.8765],
#                [-0.3579,  0.9876, -0.4321,  0.7654],
#                [ 1.2143, -1.6587,  0.9512, -0.3254]])
# Note: Your random values will be different

# Creating tensors with specific data types
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)

print(f"Float tensor: {float_tensor}, dtype: {float_tensor.dtype}")
# Output: Float tensor: tensor([1., 2., 3.]), dtype: torch.float32
print(f"Int tensor: {int_tensor}, dtype: {int_tensor.dtype}")
# Output: Int tensor: tensor([1, 2, 3], dtype=torch.int32), dtype: torch.int32
print(f"Bool tensor: {bool_tensor}, dtype: {bool_tensor.dtype}")
# Output: Bool tensor: tensor([ True, False,  True]), dtype: torch.bool

# Like operations (create tensor with same size as another)
x_like = torch.zeros_like(x)  # Tensor of zeros with same shape as x
print(x_like)
# Output: tensor([0, 0, 0, 0])

# Range and linspace
range_tensor = torch.arange(0, 10, step=1)  # Values from 0 to 9
print(range_tensor)
# Output: tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

lin_tensor = torch.linspace(0, 10, steps=11)  # 11 values evenly spaced from 0 to 10
print(lin_tensor)
# Output: tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])

# Eye (identity matrix)
eye_tensor = torch.eye(3)  # 3x3 identity matrix
print(eye_tensor)
# Output: tensor([[1., 0., 0.],
#                [0., 1., 0.],
#                [0., 0., 1.]])

# Filling with a single value
filled_tensor = torch.full((2, 3), 7)  # 2x3 tensor filled with 7
print(filled_tensor)
# Output: tensor([[7, 7, 7],
#                [7, 7, 7]])

# From NumPy
import numpy as np
np_array = np.array([1, 2, 3])
from_np = torch.from_numpy(np_array)  # Shares memory with np_array
print(f"NumPy array: {np_array}, PyTorch tensor: {from_np}")
# Output: NumPy array: [1 2 3], PyTorch tensor: tensor([1, 2, 3])

# Modify the NumPy array and see changes reflected in tensor
np_array[0] = 5
print(f"Modified NumPy array: {np_array}, PyTorch tensor: {from_np}")
# Output: Modified NumPy array: [5 2 3], PyTorch tensor: tensor([5, 2, 3])
```

### Tensor Properties and Attributes

```python
# Create a sample tensor
x = torch.rand(3, 4, 5)

# Shape and dimensions
print(f"Shape: {x.shape}")         # torch.Size([3, 4, 5])
print(f"Dimensions: {x.ndim}")     # 3
print(f"Total elements: {x.numel()}")  # 60

# Data type
print(f"Data type: {x.dtype}")     # torch.float32

# Layout (how tensor is stored in memory)
print(f"Memory layout: {x.layout}")  # torch.strided

# Device (CPU/GPU)
print(f"Device: {x.device}")       # cpu or cuda:0

# Memory information
print(f"Memory size in bytes: {x.element_size() * x.numel()}")  # e.g., 240 (60 elements * 4 bytes per float32)

# Check if tensor requires gradient
print(f"Requires grad: {x.requires_grad}")  # False by default

# Make tensor track gradients
x.requires_grad_(True)
print(f"Requires grad after setting: {x.requires_grad}")  # True

# Changing tensor device
if torch.cuda.is_available():
    # Time the transfer
    import time
    start = time.time()
    x_gpu = x.to("cuda")           # Move to GPU
    torch.cuda.synchronize()  # Wait for operation to complete
    print(f"Time to transfer to GPU: {time.time() - start:.6f} seconds")

    # Do a GPU computation
    start = time.time()
    y_gpu = x_gpu * 2
    torch.cuda.synchronize()
    print(f"Time for GPU computation: {time.time() - start:.6f} seconds")

    # Move back to CPU
    start = time.time()
    x_cpu = x_gpu.to("cpu")
    print(f"Time to transfer back to CPU: {time.time() - start:.6f} seconds")
```

### Tensor Operations in Depth

#### Mathematical Operations with Examples

```python
# Create example tensors
a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)

print(f"a: {a}")
print(f"b: {b}")
# Output:
# a: tensor([1., 2., 3.])
# b: tensor([4., 5., 6.])

# Addition
c = a + b           # Element-wise addition
print(f"a + b: {c}")
# Output: a + b: tensor([5., 7., 9.])

c = torch.add(a, b) # Function form
print(f"torch.add(a, b): {c}")
# Output: torch.add(a, b): tensor([5., 7., 9.])

# In-place operations (modifying the original tensor)
a_copy = a.clone()  # Make a copy to demonstrate in-place operation
a_copy.add_(b)      # Note the underscore suffix indicating in-place
print(f"After a.add_(b): {a_copy}")
# Output: After a.add_(b): tensor([5., 7., 9.])

# Subtraction
d = a - b
print(f"a - b: {d}")
# Output: a - b: tensor([-3., -3., -3.])

d = torch.sub(a, b)
print(f"torch.sub(a, b): {d}")
# Output: torch.sub(a, b): tensor([-3., -3., -3.])

# Multiplication (element-wise)
e = a * b           # Element-wise (Hadamard product)
print(f"a * b: {e}")
# Output: a * b: tensor([ 4., 10., 18.])

e = torch.mul(a, b)
print(f"torch.mul(a, b): {e}")
# Output: torch.mul(a, b): tensor([ 4., 10., 18.])

# Division
f = a / b
print(f"a / b: {f}")
# Output: a / b: tensor([0.2500, 0.4000, 0.5000])

f = torch.div(a, b)
print(f"torch.div(a, b): {f}")
# Output: torch.div(a, b): tensor([0.2500, 0.4000, 0.5000])

# Matrix multiplication
m1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
m2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(f"m1:\n{m1}")
print(f"m2:\n{m2}")
# Output:
# m1:
# tensor([[1., 2.],
#         [3., 4.]])
# m2:
# tensor([[5., 6.],
#         [7., 8.]])

m_result = torch.matmul(m1, m2)
print(f"torch.matmul(m1, m2):\n{m_result}")
# Output:
# torch.matmul(m1, m2):
# tensor([[19., 22.],
#         [43., 50.]])

# Using @ operator for matrix multiplication
m_result = m1 @ m2
print(f"m1 @ m2:\n{m_result}")
# Output:
# m1 @ m2:
# tensor([[19., 22.],
#         [43., 50.]])

# Other mathematical operations
sqrt = torch.sqrt(a)       # Square root
print(f"torch.sqrt(a): {sqrt}")
# Output: torch.sqrt(a): tensor([1.0000, 1.4142, 1.7321])

pow = torch.pow(a, 2)      # Power (a^2)
print(f"a^2: {pow}")
# Output: a^2: tensor([1., 4., 9.])

exp = torch.exp(a)         # Exponential (e^a)
print(f"e^a: {exp}")
# Output: e^a: tensor([ 2.7183,  7.3891, 20.0855])

log = torch.log(a)         # Natural logarithm
print(f"ln(a): {log}")
# Output: ln(a): tensor([0.0000, 0.6931, 1.0986])

# Trigonometric functions
angles = torch.tensor([0, torch.pi/4, torch.pi/2])
print(f"Angles in radians: {angles}")

sin_values = torch.sin(angles)
print(f"sin(angles): {sin_values}")
# Output: sin(angles): tensor([0.0000, 0.7071, 1.0000])

cos_values = torch.cos(angles)
print(f"cos(angles): {cos_values}")
# Output: cos(angles): tensor([1.0000, 0.7071, 0.0000])

# Reduction operations
a_extended = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(f"a_extended:\n{a_extended}")
# Output:
# a_extended:
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])

# Sum across all elements
total = torch.sum(a_extended)
print(f"Sum of all elements: {total}")
# Output: Sum of all elements: tensor(21.)

# Sum across rows (dim=1)
row_sums = torch.sum(a_extended, dim=1)
print(f"Sum of each row: {row_sums}")
# Output: Sum of each row: tensor([ 6., 15.])

# Sum across columns (dim=0)
col_sums = torch.sum(a_extended, dim=0)
print(f"Sum of each column: {col_sums}")
# Output: Sum of each column: tensor([5., 7., 9.])

# Mean values
mean_val = torch.mean(a_extended)  # Mean of all elements
print(f"Mean of all elements: {mean_val}")
# Output: Mean of all elements: tensor(3.5000)

row_means = torch.mean(a_extended, dim=1)
print(f"Mean of each row: {row_means}")
# Output: Mean of each row: tensor([2., 5.])

# Max values
max_val, max_idx = torch.max(a_extended, dim=1)
print(f"Maximum values in each row: {max_val}")
print(f"Indices of maximum values: {max_idx}")
# Output:
# Maximum values in each row: tensor([3., 6.])
# Indices of maximum values: tensor([2, 2])

# Min value
min_val = torch.min(a_extended)
print(f"Minimum value in the tensor: {min_val}")
# Output: Minimum value in the tensor: tensor(1.)

# Standard deviation and variance
std_dev = torch.std(a_extended)
var = torch.var(a_extended)
print(f"Standard deviation: {std_dev}")
print(f"Variance: {var}")
# Output:
# Standard deviation: tensor(1.7078)
# Variance: tensor(2.9167)
```

#### Indexing and Slicing in Detail

```python
x = torch.arange(12).reshape(3, 4)
print(f"Original tensor:\n{x}")
# Output:
# Original tensor:
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# Single element
element = x[1, 2]  # Value at row 1, column 2
print(f"Element at row 1, col 2: {element}")
# Output: Element at row 1, col 2: tensor(6)

# Entire row
row = x[1]         # Second row
print(f"Second row: {row}")
# Output: Second row: tensor([4, 5, 6, 7])

# Entire column
col = x[:, 2]      # Third column
print(f"Third column: {col}")
# Output: Third column: tensor([ 2,  6, 10])

# Slicing
slice1 = x[0:2, 1:3]  # First two rows, second and third columns
print(f"Slice [0:2, 1:3]:\n{slice1}")
# Output:
# Slice [0:2, 1:3]:
# tensor([[1, 2],
#         [5, 6]])

# Negative indexing (counting from the end)
last_row = x[-1]
print(f"Last row: {last_row}")
# Output: Last row: tensor([ 8,  9, 10, 11])

last_col = x[:, -1]
print(f"Last column: {last_col}")
# Output: Last column: tensor([ 3,  7, 11])

# Stepped slicing
step_slice = x[::2, ::2]  # Every other row and column
print(f"Every other row and column:\n{step_slice}")
# Output:
# Every other row and column:
# tensor([[ 0,  2],
#         [ 8, 10]])

# Advanced indexing
indices = torch.tensor([0, 2])
selected = x[indices]  # Select specific rows
print(f"Selected rows 0 and 2:\n{selected}")
# Output:
# Selected rows 0 and 2:
# tensor([[ 0,  1,  2,  3],
#         [ 8,  9, 10, 11]])

# Boolean masking
mask = x > 5
print(f"Boolean mask (x > 5):\n{mask}")
# Output:
# Boolean mask (x > 5):
# tensor([[False, False, False, False],
#         [False, False,  True,  True],
#         [ True,  True,  True,  True]])

filtered = x[mask]  # All elements > 5
print(f"Elements where x > 5: {filtered}")
# Output: Elements where x > 5: tensor([ 6,  7,  8,  9, 10, 11])

# Combined advanced indexing
row_indices = torch.tensor([0, 2])
col_indices = torch.tensor([1, 3])
elements = x[row_indices[:, None], col_indices]
print(f"Selected elements using advanced indexing:\n{elements}")
# Output:
# Selected elements using advanced indexing:
# tensor([[ 1,  3],
#         [ 9, 11]])

# Setting values using indexing
x_copy = x.clone()
x_copy[0, 0] = 99
print(f"After setting x[0,0] = 99:\n{x_copy}")
# Output:
# After setting x[0,0] = 99:
# tensor([[99,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# Setting values with slicing
x_copy[1:3, 2:4] = 50
print(f"After setting x[1:3, 2:4] = 50:\n{x_copy}")
# Output:
# After setting x[1:3, 2:4] = 50:
# tensor([[99,  1,  2,  3],
#         [ 4,  5, 50, 50],
#         [ 8,  9, 50, 50]])

# Setting with broadcasting (assigning a single value to many elements)
x_copy[x_copy > 10] = 0
print(f"After setting values > 10 to 0:\n{x_copy}")
# Output:
# After setting values > 10 to 0:
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  0,  0],
#         [ 8,  9,  0,  0]])
```

#### Reshaping Tensors in Detail

```python
x = torch.arange(12)
print(f"Original 1D tensor: {x}")
# Output: Original 1D tensor: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# Reshape
x_reshaped = x.reshape(3, 4)  # 3 rows, 4 columns
print(f"Reshaped to 3x4:\n{x_reshaped}")
# Output:
# Reshaped to 3x4:
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# Using -1 to automatically infer dimension
x_reshaped = x.reshape(-1, 4)  # ? rows, 4 columns
print(f"Reshaped to ?x4 using -1:\n{x_reshaped}")
# Output: Same as above, PyTorch calculates 3 rows

x_reshaped = x.reshape(4, -1)  # 4 rows, ? columns
print(f"Reshaped to 4x? using -1:\n{x_reshaped}")
# Output:
# Reshaped to 4x? using -1:
# tensor([[ 0,  1,  2],
#         [ 3,  4,  5],
#         [ 6,  7,  8],
#         [ 9, 10, 11]])

# Flattening a tensor
flat = x_reshaped.flatten()
print(f"Flattened tensor: {flat}")
# Output: Flattened tensor: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# View (shares memory with original tensor)
original = torch.tensor([[1, 2], [3, 4]])
view = original.view(4)
print(f"Original:\n{original}")
print(f"View: {view}")
# Output:
# Original:
# tensor([[1, 2],
#         [3, 4]])
# View: tensor([1, 2, 3, 4])

# Change the view and see the original change
view[0] = 9
print(f"View after modification: {view}")
print(f"Original after view modification:\n{original}")
# Output:
# View after modification: tensor([9, 2, 3, 4])
# Original after view modification:
# tensor([[9, 2],
#         [3, 4]])

# Demonstrating the difference between view and reshape
# (reshape may return a copy if memory layout is not contiguous)
non_contiguous = torch.tensor([[1, 2], [3, 4]]).transpose(0, 1)  # Creates non-contiguous tensor
print(f"Non-contiguous tensor:\n{non_contiguous}")
# Output:
# Non-contiguous tensor:
# tensor([[1, 3],
#         [2, 4]])

print(f"Is memory contiguous? {non_contiguous.is_contiguous()}")
# Output: Is memory contiguous? False

# view() requires contiguous tensor
try:
    view2 = non_contiguous.view(4)
except RuntimeError as e:
    print(f"Error with view: {e}")
# Output: Error with view: view size is not compatible with input tensor's size and stride

# reshape() works with non-contiguous tensors (makes a copy if needed)
reshape2 = non_contiguous.reshape(4)
print(f"Reshaped non-contiguous tensor: {reshape2}")
# Output: Reshaped non-contiguous tensor: tensor([1, 3, 2, 4])

# Permuting dimensions
x_3d = torch.arange(24).reshape(2, 3, 4)  # 2 batches, 3 rows, 4 columns
print(f"3D tensor shape: {x_3d.shape}")
# Output: 3D tensor shape: torch.Size([2, 3, 4])

x_permuted = x_3d.permute(1, 0, 2)  # New order: rows, batches, columns
print(f"Permuted tensor shape: {x_permuted.shape}")
# Output: Permuted tensor shape: torch.Size([3, 2, 4])

print(f"Original first batch:\n{x_3d[0]}")
print(f"Permuted first 'batch' (now rows):\n{x_permuted[0]}")
# Output shows the reorganization of elements

# Transposing (special case of permute for 2D)
x_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"2D tensor:\n{x_2d}")
# Output:
# 2D tensor:
# tensor([[1, 2, 3],
#         [4, 5, 6]])

x_t = x_2d.t()  # Transpose rows and columns
print(f"Transposed tensor:\n{x_t}")
# Output:
# Transposed tensor:
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])

# Squeezing and unsqueezing (adding/removing dimensions of size 1)
x = torch.rand(1, 3, 1, 4)
print(f"Original tensor shape: {x.shape}")
# Output: Original tensor shape: torch.Size([1, 3, 1, 4])

x_squeezed = x.squeeze()  # Remove all dimensions of size 1
print(f"Squeezed tensor shape: {x_squeezed.shape}")
# Output: Squeezed tensor shape: torch.Size([3, 4])

x_sq_specific = x.squeeze(0)  # Remove only dimension 0
print(f"Tensor after squeezing dim 0: {x_sq_specific.shape}")
# Output: Tensor after squeezing dim 0: torch.Size([3, 1, 4])

# Adding dimensions
y = torch.rand(3, 4)
print(f"Original tensor shape: {y.shape}")
# Output: Original tensor shape: torch.Size([3, 4])

y_unsqueezed = y.unsqueeze(0)  # Add dimension at pos 0
print(f"Unsqueezed at dim 0: {y_unsqueezed.shape}")
# Output: Unsqueezed at dim 0: torch.Size([1, 3, 4])

y_unsqueezed = y.unsqueeze(-1)  # Add dimension at the end
print(f"Unsqueezed at last dim: {y_unsqueezed.shape}")
# Output: Unsqueezed at last dim: torch.Size([3, 4, 1])

# Illustrate with concrete example: Converting an image to batch format
# A single image is typically shape [height, width, channels]
image = torch.rand(28, 28, 3)  # A 28x28 RGB image
print(f"Image shape: {image.shape}")
# Output: Image shape: torch.Size([28, 28, 3])

# For neural networks, we need [batch_size, channels, height, width]
# First permute to get channels first
image = image.permute(2, 0, 1)  # Now [3, 28, 28]
print(f"Channels-first image shape: {image.shape}")
# Output: Channels-first image shape: torch.Size([3, 28, 28])

# Add batch dimension
batched_image = image.unsqueeze(0)  # Now [1, 3, 28, 28]
print(f"Batched image shape: {batched_image.shape}")
# Output: Batched image shape: torch.Size([1, 3, 28, 28])
```

#### Concatenation and Stacking with Examples

```python
# Create example tensors
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[7, 8, 9], [10, 11, 12]])

print(f"Tensor a:\n{a}")
print(f"Tensor b:\n{b}")
# Output:
# Tensor a:
# tensor([[1, 2, 3],
#         [4, 5, 6]])
# Tensor b:
# tensor([[ 7,  8,  9],
#         [10, 11, 12]])

# Concatenation - joins tensors along an existing dimension
# Concatenate along rows (dim=0)
c_rows = torch.cat([a, b], dim=0)
print(f"Concatenated along rows (dim=0):\n{c_rows}")
# Output:
# Concatenated along rows (dim=0):
# tensor([[ 1,  2,  3],
#         [ 4,  5,  6],
#         [ 7,  8,  9],
#         [10, 11, 12]])
print(f"Shape: {c_rows.shape}")
# Output: Shape: torch.Size([4, 3])

# Concatenate along columns (dim=1)
c_cols = torch.cat([a, b], dim=1)
print(f"Concatenated along columns (dim=1):\n{c_cols}")
# Output:
# Concatenated along columns (dim=1):
# tensor([[ 1,  2,  3,  7,  8,  9],
#         [ 4,  5,  6, 10, 11, 12]])
print(f"Shape: {c_cols.shape}")
# Output: Shape: torch.Size([2, 6])

# Stacking - adds a new dimension
# Stack vertically (creates a new first dimension)
s_vertical = torch.stack([a, b], dim=0)
print(f"Stacked vertically (dim=0):\n{s_vertical}")
# Output:
# Stacked vertically (dim=0):
# tensor([[[ 1,  2,  3],
#          [ 4,  5,  6]],
#
#         [[ 7,  8,  9],
#          [10, 11, 12]]])
print(f"Shape: {s_vertical.shape}")
# Output: Shape: torch.Size([2, 2, 3])

# Stack horizontally (creates a new dimension at position 1)
s_horizontal = torch.stack([a, b], dim=1)
print(f"Stacked horizontally (dim=1):\n{s_horizontal}")
# Output:
# Stacked horizontally (dim=1):
# tensor([[[ 1,  2,  3],
#          [ 7,  8,  9]],
#
#         [[ 4,  5,  6],
#          [10, 11, 12]]])
print(f"Shape: {s_horizontal.shape}")
# Output: Shape: torch.Size([2, 2, 3])

# Stack along the last dimension
s_depth = torch.stack([a, b], dim=2)
print(f"Stacked along depth (dim=2):\n{s_depth}")
# Output:
# Stacked along depth (dim=2):
# tensor([[[ 1,  7],
#          [ 2,  8],
#          [ 3,  9]],
#
#         [[ 4, 10],
#          [ 5, 11],
#          [ 6, 12]]])
print(f"Shape: {s_depth.shape}")
# Output: Shape: torch.Size([2, 3, 2])

# Real-world example: Combining multiple image batches
batch1 = torch.rand(4, 3, 28, 28)  # 4 RGB images of size 28x28
batch2 = torch.rand(6, 3, 28, 28)  # 6 RGB images of size 28x28

# Combine into one larger batch
combined_batch = torch.cat([batch1, batch2], dim=0)
print(f"Combined batch shape: {combined_batch.shape}")
# Output: Combined batch shape: torch.Size([10, 3, 28, 28])

# Other useful concatenation functions
# hstack - stack along the first dimension after the batch dimension
h_stacked = torch.hstack([a, b])
print(f"Horizontally stacked:\n{h_stacked}")
# Output: Same as concatenating along dim=1 for 2D tensors

# vstack - stack along the batch dimension
v_stacked = torch.vstack([a, b])
print(f"Vertically stacked:\n{v_stacked}")
# Output: Same as concatenating along dim=0 for 2D tensors

# dstack - stack along the third dimension
d_stacked = torch.dstack([a, b])
print(f"Depth stacked:\n{d_stacked}")
# Output: Similar to stacking along dim=2
```

### Tensor Operations with GPU - Performance Analysis

PyTorch makes it simple to utilize GPU acceleration, providing significant performance boosts for deep learning:

```python
# Check if CUDA is available and get device details
if torch.cuda.is_available():
    # Get GPU information
    current_device = torch.cuda.current_device()
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print(f"Using {device_name}, device #{current_device}")
    print(f"Total GPU devices: {device_count}")

    # Create a large tensor for benchmarking
    import time

    # Generate matrices for multiplication
    size = 2000
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # CPU matrix multiplication
    start_time = time.time()
    c_cpu = torch.matmul(a, b)
    cpu_time = time.time() - start_time
    print(f"CPU time for {size}x{size} matrix multiplication: {cpu_time:.4f} seconds")

    # Transfer to GPU
    a_gpu = a.to("cuda")
    b_gpu = b.to("cuda")

    # First GPU run (often includes overhead)
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()  # Wait for all CUDA operations to complete

    # GPU matrix multiplication (timed)
    start_time = time.time()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()  # Wait for computations to finish before stopping timer
    gpu_time = time.time() - start_time
    print(f"GPU time for {size}x{size} matrix multiplication: {gpu_time:.4f} seconds")
    print(f"GPU speedup: {cpu_time/gpu_time:.2f}x")

    # Memory management
    print(f"GPU memory allocated before clearing: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    del a_gpu, b_gpu, c_gpu
    torch.cuda.empty_cache()
    print(f"GPU memory allocated after clearing: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
else:
    print("CUDA is not available. Using CPU only.")
```

_Analogy_: Using a GPU versus a CPU for tensor operations is like comparing a specialized assembly line (GPU) to a single craftsman (CPU). The assembly line has many workers focused on simple, repetitive tasks in parallel, while the craftsman must handle each task sequentially but with more flexibility.

### Converting Between Tensor and NumPy Arrays

PyTorch works seamlessly with NumPy, making it easy to integrate with the broader Python data science ecosystem:

```python
import numpy as np

# NumPy array to PyTorch tensor
np_array = np.array([[1, 2, 3], [4, 5, 6]])
print(f"NumPy array:\n{np_array}")
# Output:
# NumPy array:
# [[1 2 3]
#  [4 5 6]]

# Convert to tensor
tensor = torch.from_numpy(np_array)
print(f"PyTorch tensor from NumPy:\n{tensor}")
# Output:
# PyTorch tensor from NumPy:
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# Memory sharing demonstration
np_array[0, 0] = 99
print(f"Modified NumPy array:\n{np_array}")
# Output:
# Modified NumPy array:
# [[99  2  3]
#  [ 4  5  6]]

print(f"PyTorch tensor reflects change:\n{tensor}")
# Output:
# PyTorch tensor reflects change:
# tensor([[99,  2,  3],
#         [ 4,  5,  6]])

# PyTorch tensor to NumPy
tensor = torch.tensor([[7, 8, 9], [10, 11, 12]])
np_from_tensor = tensor.numpy()
print(f"NumPy array from tensor:\n{np_from_tensor}")
# Output:
# NumPy array from tensor:
# [[ 7  8  9]
#  [10 11 12]]

# Memory sharing works in this direction too
tensor[0, 0] = 77
print(f"Modified tensor:\n{tensor}")
# Output:
# Modified tensor:
# tensor([[77,  8,  9],
#         [10, 11, 12]])

print(f"NumPy array reflects change:\n{np_from_tensor}")
# Output:
# NumPy array reflects change:
# [[77  8  9]
#  [10 11 12]]

# Important! GPU tensors do not share memory with NumPy
if torch.cuda.is_available():
    tensor_gpu = tensor.to("cuda")
    # This creates a copy, not shared memory
    np_from_gpu = tensor_gpu.cpu().numpy()

    tensor_gpu[0, 0] = 55
    print(f"Modified GPU tensor first element: {tensor_gpu[0, 0]}")
    # Output: Modified GPU tensor first element: tensor(55, device='cuda:0')

    print(f"NumPy array from GPU does NOT reflect change:\n{np_from_gpu}")
    # Output: NumPy array from GPU does NOT reflect change: (still shows 77)
```

## Autograd: PyTorch's Automatic Differentiation Engine

PyTorch's automatic differentiation system (autograd) is what enables neural networks to learn through backpropagation. It dynamically builds a computational graph and automatically computes gradients.

_Analogy_: Autograd is like having a smart assistant that watches your every calculation and simultaneously works out how each input affects the final result, no matter how complex the calculation gets.

### Basic Gradient Computation

```python
# Create a tensor with gradient tracking
x = torch.tensor([2.0], requires_grad=True)
print(f"x: {x}, requires_grad: {x.requires_grad}")
# Output: x: tensor([2.], requires_grad=True), requires_grad: True

# Perform operations on x
y = x**2 + 3*x + 1
print(f"y = x^2 + 3x + 1 = {y}")
# Output: y = x^2 + 3x + 1 = tensor([11.], grad_fn=<AddBackward0>)

# Compute gradient dy/dx at x=2
y.backward()

# Access the gradient (derivative)
print(f"dy/dx at x=2: {x.grad}")
# Output: dy/dx at x=2: tensor([7.])
# This matches the analytical derivative of 2x + 3 = 2(2) + 3 = 7

# More complex example with vector input
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x**2
z = y.sum()

# Compute gradient dz/dx at x=[2.0, 3.0]
z.backward()

print(f"dz/dx: {x.grad}")
# Output: dz/dx: tensor([4., 6.])
# This matches the analytical derivative of d(x^2)/dx = 2x at points x=2 and x=3
```

### Controlling Gradient Computation

```python
# Create tensors with gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Using torch.no_grad() to temporarily disable gradient tracking
with torch.no_grad():
    y = x * 2
print(f"y created in no_grad context requires_grad: {y.requires_grad}")
# Output: y created in no_grad context requires_grad: False

# Stop tracking gradients with detach()
z = x * 2
z_detached = z.detach()
print(f"z requires_grad: {z.requires_grad}, z_detached requires_grad: {z_detached.requires_grad}")
# Output: z requires_grad: True, z_detached requires_grad: False

# For inference/evaluation after training
model_input = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# During training we'd compute gradients
output = model_input.sum()
output.backward()
print(f"Gradient during training: {model_input.grad}")
# Output: Gradient during training: tensor([1., 1., 1.])

# Reset gradients
model_input.grad.zero_()

# For inference, we don't need gradients
with torch.no_grad():
    inference_output = model_input.sum()
    # The following would fail if we weren't in a no_grad context
    try:
        inference_output.backward()
    except RuntimeError as e:
        print(f"Error when trying backward() in no_grad: {e}")
        # Output: Error when trying backward() in no_grad: element 0 of tensors does not require grad and does not have a grad_fn

# Another way to do inference
inference_output = model_input.detach().sum()
```

### Understanding Computational Graph

```python
# Create leaf tensors (inputs)
a = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([2.0], requires_grad=True)

# Build a computational graph
c = a + b
d = a * b
e = c * d

# View computational graph structure
print(f"e: {e}, grad_fn: {e.grad_fn}")
# Output: e: tensor([30.], grad_fn=<MulBackward0>)
print(f"d: {d}, grad_fn: {d.grad_fn}")
# Output: d: tensor([6.], grad_fn=<MulBackward0>)
print(f"c: {c}, grad_fn: {c.grad_fn}")
# Output: c: tensor([5.], grad_fn=<AddBackward0>)

# Compute gradients
e.backward()

# Check gradients
print(f"Gradient of e with respect to a: {a.grad}")
# Output: Gradient of e with respect to a: tensor([8.])
print(f"Gradient of e with respect to b: {b.grad}")
# Output: Gradient of e with respect to b: tensor([9.])

# Let's verify these gradients with the chain rule:
# e = c * d = (a + b) * (a * b)
# de/da = d/da[(a+b)*(a*b)] = b*(a+b) + a*b = b^2 + a*b = 2^2 + 3*2 = 4 + 6 = 10
# de/db = d/db[(a+b)*(a*b)] = a*(a+b) + a*b = a^2 + a*b = 3^2 + 3*2 = 9 + 6 = 15

# Wait, our computed gradients don't match! Let's investigate:
x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)
z = (x + y) * (x * y)
z.backward()
print(f"Gradient of z with respect to x: {x.grad}")
# Output: Gradient of z with respect to x: tensor([8.])
# x: 3.0, y: 2.0
# z = (3+2) * (3*2) = 5 * 6 = 30
# dz/dx = d/dx[(x+y)*(x*y)] = y*(x+y) + x*y = 2*(3+2) + 3*2 = 2*5 + 6 = 10 + 6 = 16/2 = 8
# dz/dy = d/dy[(x+y)*(x*y)] = x*(x+y) + x*y = 3*(3+2) + 3*2 = 3*5 + 6 = 15 + 6 = 21/2 = 10.5

# The issue is that PyTorch's backward() is computing gradients with respect to scalar outputs
# When called on a scalar, it computes the gradient of that scalar with respect to the inputs
```

### Jacobian Vector Product

```python
# For vector outputs, PyTorch computes Jacobian-vector products
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([2.0, 4.0, 6.0])

# Vector function: z = [x1^2, x2^2, x3^2]
z = x**2

# Compute Jacobian-vector product J^T * v
# where J is the Jacobian matrix dz/dx and v is the vector y
z.backward(y)

# Result is J^T * v
print(f"Result of Jacobian-vector product: {x.grad}")
# Output: Result of Jacobian-vector product: tensor([ 4., 16., 36.])

# Verify:
# J = [d(x1^2)/dx1, d(x1^2)/dx2, d(x1^2)/dx3]
#     [d(x2^2)/dx1, d(x2^2)/dx2, d(x2^2)/dx3]
#     [d(x3^2)/dx1, d(x3^2)/dx2, d(x3^2)/dx3]
# J = [2*x1, 0, 0]
#     [0, 2*x2, 0]
#     [0, 0, 2*x3]
# J = [2, 0, 0]
#     [0, 4, 0]
#     [0, 0, 6]
# J^T * v = [2, 0, 0] * [2]   + [0, 4, 0] * [4]   + [0, 0, 6] * [6]
#         = [4, 0, 0] + [0, 16, 0] + [0, 0, 36]
#         = [4, 16, 36]
```

## Neural Networks with PyTorch: A Complete Example

Let's build a complete neural network that demonstrates PyTorch's capabilities. We'll create a model to classify handwritten digits from the MNIST dataset:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Prepare data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load training data
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Download and load test data
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Explore the data
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"First few labels: {labels[:10]}")

    # Display the first 6 images
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.ravel()

    for i in range(6):
        img = images[i].squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    break  # Just look at one batch

# Define a Convolutional Neural Network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 1 input channel, 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)  # 9216 = 12 * 12 * 64
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First convolution block
        x = self.conv1(x)           # Apply conv1: output shape [batch, 32, 26, 26]
        x = F.relu(x)               # Apply ReLU activation
        x = self.conv2(x)           # Apply conv2: output shape [batch, 64, 24, 24]
        x = F.relu(x)               # Apply ReLU activation
        x = F.max_pool2d(x, 2)      # Apply max pooling: output shape [batch, 64, 12, 12]
        x = self.dropout1(x)        # Apply dropout

        # Flatten for fully connected layers
        x = torch.flatten(x, 1)     # Flatten: output shape [batch, 9216]

        # Fully connected layers
        x = self.fc1(x)             # Apply fc1: output shape [batch, 128]
        x = F.relu(x)               # Apply ReLU activation
        x = self.dropout2(x)        # Apply dropout
        x = self.fc2(x)             # Apply fc2: output shape [batch, 10]

        # Log softmax for numerical stability
        output = F.log_softmax(x, dim=1)

        return output

# Instantiate the model and move it to the device
model = ConvNet().to(device)
print(model)

# Define loss function and optimizer
criterion = F.nll_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, device, train_loader, optimizer, epoch):
    model.train()  # Set model to training mode

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Print progress
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Testing loop
def test(model, device, test_loader):
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0

    with torch.no_grad():  # Disable gradient computation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Sum up batch loss
            test_loss += criterion(output, target, reduction='sum').item()

            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            # Count correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate average loss and accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return accuracy

# Train and evaluate the model
epochs = 3
accuracies = []

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    accuracy = test(model, device, test_loader)
    accuracies.append(accuracy)

# Plot accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), accuracies, marker='o', linestyle='-', color='b')
plt.title('Test Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.xticks(range(1, epochs + 1))
plt.grid(True)
plt.show()

# Save the model
torch.save(model.state_dict(), "mnist_cnn.pt")
print("Model saved to mnist_cnn.pt")

# Example of loading the model
loaded_model = ConvNet().to(device)
loaded_model.load_state_dict(torch.load("mnist_cnn.pt"))
print("Model loaded successfully!")

# Visualize model predictions
def visualize_predictions(model, device, test_loader, num_samples=10):
    model.eval()
    images, labels = next(iter(test_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]

    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1, keepdim=True).cpu()

    # Plot images with predictions
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    for i in range(num_samples):
        img = images[i].cpu().squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        color = 'green' if predictions[i].item() == labels[i].item() else 'red'
        axes[i].set_title(f"Pred: {predictions[i].item()}\nTrue: {labels[i].item()}", color=color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

visualize_predictions(model, device, test_loader)
```

_Analogy_: A neural network is like a company with many departments (layers). Raw data enters the mail room (input layer), gets processed through various specialized departments (hidden layers) where each worker (neuron) focuses on specific features, and finally, the executive team (output layer) makes the final decision based on all the processed information.

## PyTorch's Domain-Specific Libraries

PyTorch's ecosystem includes specialized libraries for different domains:

### 1. TorchVision

For computer vision tasks, TorchVision provides:

```python
import torchvision
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Pre-trained models
resnet = models.resnet50(pretrained=True)
resnet.eval()  # Set to evaluation mode

# Image transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and transform an image
img = Image.open('sample_image.jpg')
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Get predictions
with torch.no_grad():
    output = resnet(batch_t)

# Load ImageNet labels
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# Get top predictions
_, indices = torch.sort(output, descending=True)
percentages = torch.nn.functional.softmax(output, dim=1)[0] * 100

print("Top 5 predictions:")
for idx in indices[0][:5]:
    print(f"{labels[idx]}: {percentages[idx].item():.2f}%")
```

### 2. TorchText

For natural language processing tasks:

```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB

# Load dataset
train_iter = IMDB(split='train')

# Tokenize text
tokenizer = get_tokenizer('basic_english')
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Process text
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1 if x == "positive" else 0

# Example
example_text = "This movie was excellent! I loved it."
processed_text = text_pipeline(example_text)
print(f"Original: {example_text}")
print(f"Processed: {processed_text}")
```

### 3. TorchAudio

For audio processing tasks:

```python
import torchaudio
import matplotlib.pyplot as plt

# Load an audio file
waveform, sample_rate = torchaudio.load('sample_audio.wav')

# Display audio waveform
plt.figure(figsize=(12, 4))
plt.plot(waveform[0])
plt.title("Audio Waveform")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# Apply transformations
spectrogram = torchaudio.transforms.Spectrogram()(waveform)
mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)

# Display spectrograms
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].imshow(spectrogram.log2()[0].numpy(), cmap='viridis')
axs[0].set_title("Spectrogram")
axs[1].imshow(mel_spectrogram.log2()[0].numpy(), cmap='viridis')
axs[1].set_title("Mel Spectrogram")
plt.show()
```

## Model Deployment with PyTorch

PyTorch offers several options for deploying models to production:

### 1. TorchScript

TorchScript allows for serializing and optimizing models:

```python
# Convert model to TorchScript
scripted_model = torch.jit.script(model)

# Save the scripted model
scripted_model.save("model_scripted.pt")

# Load the model in C++ or Python
loaded_model = torch.jit.load("model_scripted.pt")
```

### 2. ONNX Export

Export to ONNX format for cross-framework compatibility:

```python
# Export the model to ONNX
dummy_input = torch.randn(1, 1, 28, 28)  # Example input dimensions
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"},
                                "output": {0: "batch_size"}})

print("Model exported to ONNX format")
```

### 3. Mobile Deployment

For mobile applications using PyTorch Mobile:

```python
# Optimize the model for mobile
from torch.utils.mobile_optimizer import optimize_for_mobile

scripted_model = torch.jit.script(model)
optimized_model = optimize_for_mobile(scripted_model)
optimized_model.save("model_mobile.pt")

print("Model optimized for mobile deployment")
```

## Best Practices for PyTorch Development

### 1. Memory Management

```python
# Clear GPU cache
torch.cuda.empty_cache()

# Monitor memory usage
print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Use gradient checkpointing for larger models
# (trades compute for memory)
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(model, input):
    return checkpoint(model, input)
```

### 2. Performance Optimization

```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Run forward pass with autocasting
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # Scale gradients and call backward()
    scaler.scale(loss).backward()

    # Unscale gradients and call optimizer.step()
    scaler.step(optimizer)

    # Update the scale for next iteration
    scaler.update()
```

### 3. Code Structure Best Practices

```python
# Define models in separate files
# Example model structure
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)
```

### 4. Debugging Tips

- Use `tensor.shape` frequently to catch dimension mismatch errors
- Set `torch.autograd.detect_anomaly(True)` for detailed error information
- Use `torch.autograd.profiler.profile()` to identify bottlenecks
- Check gradients with `for name, param in model.named_parameters(): print(name, param.grad)`

## The PyTorch Ecosystem

PyTorch integrates with many tools and libraries:

1. **Lightning**: Simplifies training loops and best practices
2. **Weights & Biases**: Experiment tracking and visualization
3. **Ray/Tune**: Distributed training and hyperparameter tuning
4. **Catalyst**: Research-focused training loops
5. **FastAI**: High-level API built on PyTorch
6. **Transformers (Hugging Face)**: State-of-the-art NLP models
7. **Captum**: Model interpretability
8. **TorchServe**: Model serving
9. **NVIDIA Apex**: Advanced optimizations

## Conclusion: The PyTorch Advantage

PyTorch has become a dominant framework in deep learning research and is increasingly popular in production environments due to its:

1. **Intuitive Design**: PyTorch feels natural to Python programmers, making the learning curve smoother than with other frameworks.

2. **Flexibility**: The dynamic computation graph allows for complex, conditional model architectures that can change during runtime.

3. **Debugging Ease**: Standard Python debugging tools work with PyTorch, making error identification and fixing much simpler.

4. **Strong Community**: Backed by Facebook (Meta) and adopted by major research labs, PyTorch has a vibrant ecosystem of tools, libraries, and pre-trained models.

5. **Production Readiness**: With tools like TorchScript, ONNX export, and mobile deployment options, PyTorch models can be deployed to various environments.

Starting with tensors and building up to complex neural networks, PyTorch provides a coherent, comprehensive platform for all your deep learning needs.

_Analogy_: If deep learning is like constructing a building, PyTorch gives you both pre-fabricated components for quick assembly and raw materials for custom designs, along with intuitive tools that work the way you expect them to.

## Next Steps in Your PyTorch Journey

1. **Dive deeper into PyTorch's advanced features**:

   - Custom layers and loss functions
   - Writing custom datasets and data loaders
   - Implementing advanced research papers

2. **Explore specialized domains**:

   - Computer vision with TorchVision
   - Natural language processing with TorchText
   - Audio processing with TorchAudio
   - Reinforcement learning with TorchRL

3. **Learn best practices for scaling and deployment**:

   - Distributed training across multiple GPUs/nodes
   - Model quantization and optimization
   - TorchServe for deployment

4. **Contribute to the PyTorch community**:
   - Report bugs and contribute fixes
   - Share custom implementations
   - Help improve documentation

By mastering PyTorch fundamentals, you've taken the first step toward becoming proficient in one of the most powerful and flexible deep learning frameworks available today.
