# fire-gpt
![](https://pbs.twimg.com/media/FpMNyPdWYAQpjw8?format=jpg&name=4096x4096)

## gelu

```py
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
```

- "When the Approximation option is "tanh", the software approximates the error function using `erf(x√2)≈tanh(√2π(x+0.044715x3))`" [1]
- The GELU nonlinearity weights inputs by their percentile, rather than gates inputs by their sign as in ReLUs. Consequently the GELU can be thought of as a smoother ReLU. [3]


```py
>>> def gelu(x):
...     return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
...
>>> gelu
<function gelu at 0x105600670>
>>> import numpy as np
>>> gelu(np.array([[1, 2],[-2, 0.5]]))
array([[ 0.84119199,  1.95459769],
       [-0.04540231,  0.34571401]])
```

- "GELU operates element-wise on the input" [0]

0. https://jaykmody.com/blog/gpt-from-scratch/
1. https://www.mathworks.com/help/deeplearning/ref/dlarray.gelu.html
2. https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions
3. https://paperswithcode.com/method/gelu
