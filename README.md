# PyTorch


![Pytorch_logo](https://user-images.githubusercontent.com/58439868/133604410-de7e686c-c683-4239-8020-616b8cc304cb.png)


>[PyTorch](https://pytorch.org/)

>[PyTorch GitHub](https://github.com/pytorch/pytorch)



```python
import torch
```


```python
x = torch.empty(2)
```
> tensor([1.8331e-40, 0.0000e+00])




```python
x = torch.rand(2)
```
>tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 4.2531e-05, 1.0802e-05]])



```python
x = torch.zeros(2)
```
>tensor([0., 0.])


```python
x = torch.ones(2)
```
>tensor([1., 1.])



```python
x = torch.ones(2)
x.dtype
```
>torch.float32

```python
x = torch.ones(2,4)
x.size()
```
>torch.Size([2, 4])

```python
x = torch.tensor([2.5,0.1])
x
```
>tensor([2.5000, 0.1000])

```python
y=torch.tensor([4.9,4.3])
y
```
>tensor([4.9000, 4.3000])

```python
z = x+y
```
>tensor([7.4000, 4.4000])

```python
y.add_(x)
```
>tensor([7.4000, 4.4000])

```python
x.add_(y)
```
>tensor([9.9000, 4.5000])

```
#underscore used to inplace operation 
```python
x
```
>tensor([9.9000, 4.5000])
```python
y
```
>tensor([7.4000, 4.4000]) 

```python
z= torch.sub(x,y)
```
>tensor([2.5000, 0.1000])

```python
x.sub_(y)
```
> tensor([2.5000, 0.1000])

```python
x = torch.rand(5,4)
x
```
>tensor([[0.3094, 0.3055, 0.9537, 0.1301],
        [0.4614, 0.6939, 0.5546, 0.1630],
        [0.9947, 0.1444, 0.0529, 0.8653],
        [0.9795, 0.6218, 0.5568, 0.8080],
        [0.5672, 0.0596, 0.5012, 0.3082]])
        
```python
x[:,0]
```
> tensor([0.3094, 0.4614, 0.9947, 0.9795, 0.5672])

```python
x[0,:]
```
> tensor([0.3094, 0.3055, 0.9537, 0.1301])

```python
x[1,1].item()
```
> 0.6939277052879333

```python
#use item() if you have single value it will print seprate value pure number

#reshaping

x=torch.rand(4,4)
x
```
> tensor([[0.0519, 0.0100, 0.5350, 0.1515],
        [0.8415, 0.6998, 0.0026, 0.1225],
        [0.7409, 0.2396, 0.6612, 0.0884],
        [0.8749, 0.2309, 0.2504, 0.6981]])
        

```python
x.view(16)
```
>tensor([0.0519, 0.0100, 0.5350, 0.1515, 0.8415, 0.6998, 0.0026, 0.1225,
          0.7409,0.2396, 0.6612, 0.0884, 0.8749, 0.2309, 0.2504, 0.6981])
        
        
```python
x.view(-1,8)
```
> tensor([[0.0519, 0.0100, 0.5350, 0.1515, 0.8415, 0.6998, 0.0026, 0.1225],
          [0.7409, 0.2396, 0.6612, 0.0884, 0.8749, 0.2309, 0.2504, 0.6981]])


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



```
#numpy to tensor and vice versa
```
```python
import numpy as np
```

```python
a = torch.ones(5)
a
```
>tensor([1., 1., 1., 1., 1.])

```python
b=a.numpy()
type(b)
```
> numpy.ndarray

```python
type(a)
```
> torch.Tensor

```python
a = np.ones(5)
a
```
> array([1., 1., 1., 1., 1.])

```python
b = torch.from_numpy(a)
b
```
> tensor([1., 1., 1., 1., 1.], dtype=torch.float64)

```python
'gpu' if torch.cuda.is_available() else 'cpu'

```























