# PyTorch


![Pytorch_logo](https://user-images.githubusercontent.com/58439868/133604410-de7e686c-c683-4239-8020-616b8cc304cb.png)


>[PyTorch](https://pytorch.org/)

>[GitHub](https://github.com/pytorch/pytorch)



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

```
x = torch.ones(2,4)
x.size()
```
```
x = torch.tensor([2.5,0.1])
x
```
```
y=torch.tensor([4.9,4.3])
y
```
```
z = x+y
z
```
```
y.add_(x)
```
```
x.add_(y)
#underscore used to inplace operation 
```
x

y
```
z= torch.sub(x,y)

z
```
```
x.sub_(y)
```
```
x = torch.rand(5,4)
x
```
```
x[:,0]
```
```
x[0,:]
```
```
x[1,1].item()
```
```
#use item() if you have single value it will print seprate value pure number

#reshaping

x=torch.rand(4,4)
x
```
```
x.view(16)
```
```
x.view(-1,8)
```
```
x.view(8,2)
```
```
x.view(8,-1)
```
```
#numpy to tensor and vice versa
```
```
import numpy as np
```
```
a = torch.ones(5)
a
```
```
b=a.numpy()
```
```
type(a)
```
```
type(b)
```


```
a = np.ones(5)
a
```
```
b = torch.from_numpy(a)
b
```
```
'gpu' if torch.cuda.is_available() else 'cpu'

```



































































































































































