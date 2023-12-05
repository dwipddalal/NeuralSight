## Gaussian Filter Function

The `gaussian` function defined in the code snippet is used to create a one-dimensional Gaussian filter kernel. This filter is commonly used in image processing for blurring and smoothing images. The function is defined with two parameters:

- `kernel_size`: The size of the kernel (default is 3).
- `sigma`: The standard deviation of the Gaussian distribution (default is calculated based on the kernel size).

### Mathematical Background

The Gaussian function, which forms the basis of this filter, is defined by the formula:

$ G(x) = e^{-\frac{1}{2} \left( \frac{x}{\sigma} \right)^2} $

$ \sigma $ is the standard deviation of the Gaussian distribution.
