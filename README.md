# PyTorch


![Pytorch_logo](https://user-images.githubusercontent.com/58439868/133604410-de7e686c-c683-4239-8020-616b8cc304cb.png)



>[PyTorch](https://pytorch.org/)

>[PyTorch GitHub](https://github.com/pytorch/pytorch)

## PyTorch libraries 

### PyTorch [Installation](https://pytorch.org/get-started/locally/)

>Libraries

PyTorch         |      Use
--------------- |---------------
torch           |     nn (Neural Network)
torchaudio      |     Audio Processing
torchvision     |     Images Processing


## Demo
- Genral PyTorch Installation Using pip/conda:
```python
pip install torch
pip install torchvision
pip install torchaudio
```
- Importing torch library

```python
import torch
```
- 1D Empty tensor filled with uninitialized data

```python
x = torch.empty(2)
```
> tensor([1.8331e-40, 0.0000e+00])


- Random tensor

```python
x = torch.rand(2)
```
>tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 4.2531e-05, 1.0802e-05]])

- Zeros tensor

```python
x = torch.zeros(2)
```
>tensor([0., 0.])

- Ones tensor
```python
x = torch.ones(2)
```
>tensor([1., 1.])

- Check data type

```python
x = torch.ones(2)
x.dtype
```
>torch.float32

- Tensor Shape and Size
```python
x = torch.ones(2,4)
x.size()
```
>torch.Size([2, 4])

- Define Tensors
```python
x = torch.tensor([2.5,0.1])
```
>tensor([2.5000, 0.1000])

```python
y=torch.tensor([4.9,4.3])
```
>tensor([4.9000, 4.3000])

- Operation's on Tensor
- Adding Tensor x and y
```python
z = x+y
```
>tensor([7.4000, 4.4000])


```python
y.add_(x)
```
>tensor([7.4000, 4.4000])

- Underscore uses for Inplace operation
```python
x.add_(y)
```
>tensor([9.9000, 4.5000])


```python
x
```
>tensor([9.9000, 4.5000])
```python
y
```
>tensor([7.4000, 4.4000]) 

```python
x.sub_(y)
```
> tensor([2.5000, 0.1000])

- Built-in operation
```python
z= torch.sub(x,y)
```
>tensor([2.5000, 0.1000])

- Random 2D Tensor

```python
x = torch.rand(5,4)
x
```
>tensor([[0.3094, 0.3055, 0.9537, 0.1301],
        [0.4614, 0.6939, 0.5546, 0.1630],
        [0.9947, 0.1444, 0.0529, 0.8653],
        [0.9795, 0.6218, 0.5568, 0.8080],
        [0.5672, 0.0596, 0.5012, 0.3082]])
   
  
- Slicing operation
```python
x[:,0]
```
> tensor([0.3094, 0.4614, 0.9947, 0.9795, 0.5672])


```python
x[0,:]
```
> tensor([0.3094, 0.3055, 0.9537, 0.1301])

- item() used for pure data representation
```python
x[1,1].item()
```
> 0.6939277052879333

- Random 4 by 4 Matrix
```python
x=torch.rand(4,4)
x
```
> tensor([[0.0519, 0.0100, 0.5350, 0.1515],
        [0.8415, 0.6998, 0.0026, 0.1225],
        [0.7409, 0.2396, 0.6612, 0.0884],
        [0.8749, 0.2309, 0.2504, 0.6981]])
        

- Reshaping Matrix
- We convert 4x4 matrix to 1D Tensor
```python
x.view(16)
```
>tensor([0.0519, 0.0100, 0.5350, 0.1515, 0.8415, 0.6998, 0.0026, 0.1225,
          0.7409,0.2396, 0.6612, 0.0884, 0.8749, 0.2309, 0.2504, 0.6981])
        
- Reshaping Matrix 4x4 to 8x2  
```python
x.view(-1,8)
```
> tensor([[0.0519, 0.0100, 0.5350, 0.1515, 0.8415, 0.6998, 0.0026, 0.1225],
          [0.7409, 0.2396, 0.6612, 0.0884, 0.8749, 0.2309, 0.2504, 0.6981]])

- Reshaping Matrix 4x4 to 2x8
```python
x.view(8,2)
```
>tensor([[0.0519, 0.0100],
        [0.5350, 0.1515],
        [0.8415, 0.6998],
        [0.0026, 0.1225],
        [0.7409, 0.2396],
        [0.6612, 0.0884],
        [0.8749, 0.2309],
        [0.2504, 0.6981]])



- Reshaping Matrix 4x4 to 2x8
```python
x.view(8,-1)
```
> tensor([[0.0519, 0.0100],
        [0.5350, 0.1515],
        [0.8415, 0.6998],
        [0.0026, 0.1225],
        [0.7409, 0.2396],
        [0.6612, 0.0884],
        [0.8749, 0.2309],
        [0.2504, 0.6981]])




- Converting Tensor to Numpy and Vice-Versa

Importing numpy

```python
import numpy as np
```

```python
a = torch.ones(5)
a
```
>tensor([1., 1., 1., 1., 1.])

- Convert Tensor to Array
```python
b = a.numpy()
type(b)
```
> numpy.ndarray

- Tensor data type
```python
type(a)
```
> torch.Tensor

- Convert Numpy Array to Tensor

```python
b = torch.from_numpy(a)
b
```
> tensor([1., 1., 1., 1., 1.], dtype=torch.float64)


- Check if you have CUDA is available for GPU operation
```python
'gpu' if torch.cuda.is_available() else 'cpu'

```
..



