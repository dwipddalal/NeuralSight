## Gaussian Filter Function

The `gaussian` function defined in the code snippet is used to create a one-dimensional Gaussian filter kernel. This filter is commonly used in image processing for blurring and smoothing images. The function is defined with two parameters:

- `kernel_size`: The size of the kernel (default is 3).
- `sigma`: The standard deviation of the Gaussian distribution (default is calculated based on the kernel size).

### Mathematical Background

The Gaussian function, which forms the basis of this filter, is defined by the formula:

$$ G(x) = e^{-\frac{1}{2} \left( \frac{x}{\sigma} \right)^2} $$

$\sigma$ is the standard deviation of the Gaussian distribution.

### `blur` Function
**Purpose**: Applies different types of blur to an image.

- For each color channel in the image:
  - Depending on the `mode`, it selects a type of blur:
    - **Box Blur**: Uniformly averages pixel values.
    - **Gaussian Blur**: Uses a Gaussian kernel for a smoother, more natural blur.
    - **Motion Blur**: Simulates the effect of movement over the image.
  - Applies the selected blur using 2D convolution.

- **Box Blur**: Each pixel is the average of its neighborhood.
- **Gaussian Blur**: Weights pixels based on their distance from the center, following a Gaussian distribution.
- **Motion Blur**: Pixels along a line are averaged, mimicking motion.

### `add_gaussian_noise` Function
**Purpose**: Adds Gaussian noise to an image.

- Generates Gaussian noise with a specified standard deviation (`sigma`).
- Adds this noise to the original image.
- Ensures the resulting pixel values stay within valid image range (0-255).

- Gaussian noise is characterized by its mean (0 in this case) and standard deviation (`sigma`), forming a normal distribution.


## The wiener filter

The Wiener filter is a digital filtering technique used in image and signal processing. It is named after Norbert Wiener and is used to produce an estimate of a desired or target random process by linear time-invariant (LTI) filtering of an observed noisy process. The main purpose of the Wiener filter is to reduce the amount of noise present in a signal by comparison with an estimate of the desired noiseless signal.

---

Let's go through the mathematical proof of why the complex conjugate appears in the implementation of the Wiener filter. In the context of complex numbers, division by a complex number involves its conjugate. Let's denote a complex number by $z = a + bi$, where $a$ is the real part, $b$ is the imaginary part, and $i$ is the imaginary unit. The complex conjugate of $z$, denoted by $\overline{z}$, is $a - bi$.

Now, to divide one complex number by another, say $\frac{z_1}{z_2}$, we multiply the numerator and denominator by the complex conjugate of the denominator:

$\frac{z_1}{z_2} = \frac{z_1}{z_2} \cdot \frac{\overline{z_2}}{\overline{z_2}} = \frac{z_1 \overline{z_2}}{z_2 \overline{z_2}} $

Since $z_2 \overline{z_2} = |z_2|^2$ (the magnitude squared of $z_2$), the division is:

$\frac{z_1}{z_2} = \frac{z_1 \overline{z_2}}{|z_2|^2} $

In the context of the Wiener filter, $G(u,v)$ and $H(u,v)$ are complex numbers representing the Fourier transforms of the image and the blur kernel, respectively. The Wiener filter formula in the frequency domain is:

$F'(u, v) = \frac{G(u, v)}{H(u, v)} \cdot \frac{1}{1 + \frac{NSR(u, v)}{|H(u, v)|^2}} $

Breaking it down:

1. $\frac{G(u, v)}{H(u, v)}$ is a division of complex numbers, which we established is implemented by multiplication with the conjugate:

$\frac{G(u, v)}{H(u, v)} = G(u, v) \cdot \frac{\overline{H(u, v)}}{|H(u, v)|^2} $

2. The Wiener filter also involves the noise-to-signal ratio $NSR(u, v)$:

$F'(u, v) = G(u, v) \cdot \frac{\overline{H(u, v)}}{|H(u, v)|^2} \cdot \frac{1}{1 + \frac{NSR(u, v)}{|H(u, v)|^2}} $

3. Simplifying the expression:

$F'(u, v) = G(u, v) \cdot \frac{\overline{H(u, v)}}{|H(u, v)|^2 + NSR(u, v)} $

In the code:

```python
kernel_fft = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
```

`kernel_fft` is $H(u,v)$, and `np.conj(kernel_fft)` computes $\overline{H(u,v)}$. The term $K$ is used as an approximation for $NSR(u, v)$.

Hence, the code is a practical implementation of the division component of the Wiener filter, assuming $G(u,v)$ has already been divided by $H(u,v)$, and `K` is used as an estimate for the noise-to-signal power ratio across all frequencies.
"""


