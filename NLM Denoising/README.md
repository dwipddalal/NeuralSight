## `Peak Signal-to-Noise Ratio (PSNR)` Formula

The PSNR is a metric used to measure the quality of reconstructed images. It is calculated using the following formula:

$$
\text{PSNR} = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{MSE} \right)
$$

where $MAX_I$ is the maximum possible pixel value of the image. For example, if all pixels are represented using 8 bits, then $MAX_I = 255$.


# Non-Local Means (NLM) Algorithm for Image Denoising

## Algorithm Overview

The NLM method is applied to a grayscale image $I$ to denoise a pixel $p$. The denoised pixel $Ib(p)$ is calculated as follows:

$$
Ib(p) = \frac{1}{C(p)} \sum_{q \in B(p, r)} I(q) w(p, q) \quad \text{(1)}
$$

Where $C(p) = \sum_{q \in B(p, r)} w(p, q)$ and $B(p, r)$ is a research window indicating a neighborhood centered at $p$ with the size $(2r + 1) \times (2r + 1)$ pixels.

### Weight Calculation

The weights $w(p, q)$ are calculated using the exponential kernel as:

$$
w(p, q) = e^{\frac{-\max(d^2 - 2\sigma^2, 0)}{h^2}} \quad \text{(2)}
$$

Here $\sigma$ is the standard deviation of the additive zero-mean Gaussian noise, and $h$ is the filtering parameter, which is a function of $\sigma$.

### Squared Euclidean Distance

The squared Euclidean distance $d^2$ is given by:

$$
d^2(B(p, f), B(q, f)) = \frac{1}{(2f + 1)^2} \sum_{j \in B(0, f)} (I(p + j) - I(q + j))^2 \quad \text{(3)}
$$

### Special Consideration

The weight of the reference pixel $p$ in the average is set to the maximum of the weights in the neighborhood $B(p, r)$. This setting avoids excessive weighting of the reference point in the average.

# Results

<img width="641" alt="image" src="https://github.com/dwipddalal/NeuralSight/assets/91228207/f425bc87-5230-408a-8b9f-fe65e60edff3">
<img width="640" alt="image" src="https://github.com/dwipddalal/NeuralSight/assets/91228207/4db66f76-74bb-4561-af45-e86321a181bc">
<img width="637" alt="image" src="https://github.com/dwipddalal/NeuralSight/assets/91228207/82dc3ae2-4545-4675-b259-184759de29fa">
<img width="641" alt="image" src="https://github.com/dwipddalal/NeuralSight/assets/91228207/2c7a5801-1b46-4162-ae9b-04b6f378182d">






# Observations on NLM Image Denoising for Different Parameters

## Experimental Setup
We denoised images using the Non-Local Means (NLM) algorithm for every $\sigma \in \{ 15, 45, 80 \}$ and tuned the parameter $h = k\sigma$ with $0.01 \leq k \leq 0.65$. We also experimented with different sizes of comparison windows $(2f + 1) \in \{ 3, 5, 7, 9 \}$.

## Observations

### Effect of $\sigma$
- **Low $\sigma$ (15)**: In the case of lower noise levels ($\sigma=15$), even a relatively simpler model with smaller $h$ and $(2f+1)$ gives satisfactory results.
- **High $\sigma$ (80)**: For a highly noisy image ($\sigma=80$), a more complex model with larger $h$ and $(2f+1)$ is needed for effective denoising.

### Effect of $h$ (Filtering Parameter)
- **Lower $h$**: Produces a less smooth image but retains more details. Risk of not removing enough noise.
- **Higher $h$**: Produces a smoother image but may result in loss of important details due to excessive blurring.

### Effect of $(2f + 1)$ (Comparison Window Size)
- **Smaller $(2f + 1)$**: Better at preserving high-frequency details but might be less effective in removing noise.
- **Larger $(2f + 1)$**: More effective in noise removal but may result in a loss of texture and finer details.

## Summary

1. Higher $\sigma$ values require a more complex model with higher $h$ and $(2f+1)$ for effective denoising.
2. Lower $h$ values are better for preserving details but are less effective at noise removal.
3. Larger $(2f+1)$ values are good for noise removal but may result in loss of texture and finer details.
