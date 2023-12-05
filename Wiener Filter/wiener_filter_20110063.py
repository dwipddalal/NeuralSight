import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from PIL import Image
import cv2

"""## Gaussian Filter Function

The `gaussian` function defined in the code snippet is used to create a one-dimensional Gaussian filter kernel. This filter is commonly used in image processing for blurring and smoothing images. The function is defined with two parameters:

- `kernel_size`: The size of the kernel (default is 3).
- `sigma`: The standard deviation of the Gaussian distribution (default is calculated based on the kernel size).

### Mathematical Background

The Gaussian function, which forms the basis of this filter, is defined by the formula:

$ G(x) = e^{-\frac{1}{2} \left( \frac{x}{\sigma} \right)^2} $

$ \sigma $ is the standard deviation of the Gaussian distribution.
"""

def gaussian(kernel_size=3, sigma=None):
    if sigma is None:
        sigma = kernel_size / 3
    ax = np.arange(-int(kernel_size / 2), int(kernel_size / 2) + 1, dtype=np.float32)
    gaussian_1d = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    gaussian_1d /= gaussian_1d.sum()
    return gaussian_1d

"""### `blur` Function
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
"""

def blur(img, mode='box', k_size=3):

    blurred = np.zeros_like(img)
    for i in range(3):
        if mode == 'box':
            h = np.ones((k_size, k_size)) / (k_size ** 2)
        elif mode == 'gaussian':
            h = gaussian(k_size).reshape(k_size, 1)
            h = np.dot(h, h.T)
            h /= h.sum()
        elif mode == 'motion':
            h = np.eye(k_size) / k_size
        blurred[:,:,i] = convolve2d(img[:,:,i], h, mode='same', boundary='wrap')
    return blurred

def add_gaussian_noise(img, sigma):

    gaussian_noise = np.random.normal(0, sigma, img.shape)
    noisy_image = img + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image

"""## Q1"""

noisy_blurred_images = []
sigmas = [0.2, 0.4, 0.6, 0.8, 1]
kernel_size = [9, 11, 19, 27, 17]

img_path = '/home/synth/Synthetic/StableDiffusionv2/doc/assignment3/Dataset/Dataset/image1.png'
img = np.array(Image.open(img_path).convert('RGB'))
Main_img = img.copy()
for j in range(len(sigmas)):

    noisy_img = add_gaussian_noise(img, sigma=sigmas[j])
    blurred_img = blur(noisy_img, mode='gaussian', k_size=kernel_size[j])
    noisy_blurred_images.append(blurred_img)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, ax in enumerate(axes):

    ax.imshow(noisy_blurred_images[i].astype(np.uint8))
    ax.set_title(f"Sigma: {sigmas[i]}, Kernel: {kernel_size[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()

"""## Q2"""

def apply_defocus_blur(image, kernel_size=(9, 9)):
    return cv2.GaussianBlur(image, kernel_size, cv2.BORDER_DEFAULT)

def add_gaussian_noise(image, mean=0, sigma=2):
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = cv2.add(image, gauss.astype('uint8'))
    return noisy_image

img = np.array(Image.open(img_path).convert('RGB'))
blurred_image = apply_defocus_blur(img, kernel_size=(9, 9))

noisy_image = add_gaussian_noise(blurred_image, mean=0, sigma=0.2)

plt.figure(figsize=(10, 10))
plt.imshow(noisy_image)
plt.axis('off')

def calculate_mse(image1, image2):
    """Calculates the Mean Squared Error (MSE) between two images."""
    # Ensure images are in floating-point format
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # Calculate squared error for each channel (for color images)
    squared_error = np.square(image1 - image2)

    # Calculate the mean squared error over all channels
    mse = np.mean(squared_error)

    return mse

print(f"MSE between our and library image: {calculate_mse(noisy_blurred_images[0], noisy_image)}")

"""## Q3

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

def wiener_filter(img_, kernel, K):
    kernel /= kernel.sum()
    Image_copy = img_.copy()
    for i in range(3):
        Image_copy_fft = fft2(Image_copy[:,:,i])
        kernel_fft = fft2(kernel, s=img_.shape[:2])
        kernel_fft = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
        Image_copy[:,:,i] = np.abs(ifft2(Image_copy_fft * kernel_fft))
    return Image_copy

def gaussian_kernel(kernel_size=3):
    h = gaussian(kernel_size).reshape(kernel_size, 1)
    h = np.dot(h, h.T)
    h /= h.sum()
    return h

def display_images(original_img, noisy_blurred_img, filter_img):
    """
    Displays the original, noisy + blurred, and filtered images side by side for comparison.

    Parameters:
    - original_img: The original image.
    - noisy_blurred_img: The image after applying noise and blur.
    - filter_img: The image after applying a filtering technique (e.g., Wiener filter).
    """
    # Display the images
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(original_img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(noisy_blurred_img.astype('uint8'))
    axs[1].set_title('Noisy + Blurred Image')
    axs[1].axis('off')

    axs[2].imshow(filter_img.astype('uint8'))
    axs[2].set_title('Wiener Filtered Image')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

for i in range(1,5):
    path = '/home/synth/Synthetic/StableDiffusionv2/doc/assignment3/Dataset/Dataset/image'
    path = path + str(i) + '.png'
    img = np.array(Image.open(path).convert('RGB'))
    for j in range(len(sigmas)):
        kernel = gaussian_kernel(kernel_size[j])
        noisy_img = add_gaussian_noise(img, sigma=sigmas[j])
        blurred_img = blur(noisy_img, mode='gaussian', k_size=kernel_size[j])
        output_img = wiener_filter(blurred_img, kernel, K=0.05)
        display_images(img, blurred_img, output_img)

        def psnr(img1, img2):
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return 100
            PIXEL_MAX = 255.0
            return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

        print(f"PSNR between our and original image: {psnr(output_img, img)}")
        print(f'PSNR between original and noisy image: {psnr(img, blurred_img)}')


        output_img = np.zeros_like(img)

"""
### `apply_filters` Function:

- **Inputs**:
  - `image`: The noisy and possibly blurred image to be processed.
  - `main_img`: The original, unaltered image used for PSNR comparison.

- **Processing Steps**:
  1. **Preprocessing**:
      - Converts the image to 8-bit and changes color space from RGB to BGR (as OpenCV uses BGR).
  
  2. **Applying Filters**:
      - **Non-local Means Filter**: Reduces noise while trying to preserve edges.
      - **Bilateral Filter**: Reduces noise with edge preservation.
      - **Median Filter** (using PIL): Reduces noise, particularly effective against salt-and-pepper noise.
      - **Gaussian Smoothing Filter** (using PIL): Smoothens the image.
      - **Wiener Filter**: Deblurs the image assuming a known degradation function (gaussian kernel used here).
  
  3. **Saving Filtered Images**:
      - Each filtered image is saved to disk.

- **PSNR Calculation**:
  - Calculates and prints the PSNR for each filtered image compared to the original image (`main_img`).

### PSNR Function:

- Calculates the PSNR between two images (`img1` and `img2`).
- The PSNR is used to quantify the reconstruction quality of the filtered images compared to the original.
- Higher PSNR values indicate better quality."""

import cv2
from PIL import Image, ImageFilter

def apply_filters(image, main_img, number):

    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Apply Non-local Means Filter
    non_local_means = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Apply Bilateral Filter
    bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)

    # Convert the OpenCV image format to PIL format for Median and Gaussian Filters
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Apply Median Filter
    median_filter = image_pil.filter(ImageFilter.MedianFilter(size=5))

    # Apply Gaussian Smoothing Filter
    gaussian_filter = image_pil.filter(ImageFilter.GaussianBlur(radius=5))

    # Apply wiener filter
    kernel = gaussian_kernel(17)
    output_img = wiener_filter(image, kernel, K=0.05)

    cv2.imwrite('original_{number}.jpg', image)
    cv2.imwrite('non_local_means_{number}.jpg', non_local_means)
    cv2.imwrite('bilateral_filter_{number}.jpg', bilateral_filter)
    cv2.imwrite('wiener_filter_{number}.jpg', output_img)
    median_filter.save('median_filter_{number}.jpg')
    gaussian_filter.save('gaussian_filter_{number}.jpg')


    images = [image, non_local_means, bilateral_filter, output_img,
          np.array(median_filter), np.array(gaussian_filter)]
    titles = ['Original', 'Non Local Means', 'Bilateral', 'Wiener', 'Median', 'Gaussian']

    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        plt.subplot(1, 6, i + 1)
        if i <4:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # Hide tick marks

    plt.show()

    ## psnr
    def psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    print("PSNR for original : ", psnr(main_img, image))
    print("PSNR for non local means: ", psnr(main_img, non_local_means))
    print("PSNR for bilateral filter: ", psnr(main_img, bilateral_filter))
    print("PSNR for median filter: ", psnr(main_img, median_filter))
    print("PSNR for gaussian filter: ", psnr(main_img, gaussian_filter))
    print("PSNR for wiener filter: ", psnr(main_img, output_img))

apply_filters(noisy_blurred_images[4], Main_img, 0)

"""## Q4"""

### for all images in the dataset
for i in range(1,5):
    path = '/home/synth/Synthetic/StableDiffusionv2/doc/assignment3/Dataset/Dataset/image'
    path = path + str(i) + '.png'
    img = np.array(Image.open(path).convert('RGB'))

    kernel = gaussian_kernel(17)
    noisy_img = add_gaussian_noise(img, sigma=1)
    blurred_img = blur(noisy_img, mode='gaussian', k_size=17)
    apply_filters(blurred_img, img, i)

"""PSNR for Original (28.54 dB): This acts like a baseline PSNR, against a blurred and noised version of the image. The PSNR of an original image against itself would be infinite (or undefined), but if this is measured against a degraded version of the image, it represents the initial degradation level.

PSNR for Non-Local Means (28.49 dB): A slight improvement over the baseline. Non-local means filter is effective in denoising while preserving edge details.

PSNR for Bilateral Filter (28.52 dB): Almost similar to the baseline, indicating minor or no significant improvement. The bilateral filter is good for edge-preserving smoothing.

PSNR for Median Filter (29.59 dB): This shows a more significant improvement. Median filters are good at removing certain types of noise like salt-and-pepper/gaussain noise.

PSNR for Gaussian Filter (29.24 dB): Indicates improvement, but not as much as the median filter. Gaussian filters are good for general blurring and noise reduction but might not preserve edges as well.

PSNR for Wiener Filter (28.13 dB): This is slightly below the baseline, suggesting that in this case, the Wiener filter might not have been as effective as other methods. Wiener filters are typically used for image deblurring. Since here the image taken is very noisy image hence wiener filter might not have given good result.


---

Another important insight is that with increase in the blurring value, the diffence in the PSNR value because the (Original image and Deblurred Image) and (Original image and blurred + noised image) increase. Showcasing the importance of weiner filter for deblurring denoising task.

---

Non-Local Means Filter: This algorithm works by comparing all patches in the image and averaging similar ones to produce the denoised image. It's particularly good at preserving detailed textures, which might explain the slight improvement in PSNR. However, it can sometimes preserve noise as texture, which might be why the improvement is marginal.

Bilateral Filter: It is designed to smooth images while preserving edges, by considering both the spatial proximity and the intensity similarity when averaging pixels. This dual consideration usually preserves edges better than a standard Gaussian blur, which might be why its PSNR is close to that of the original degradation. It provides a balance between noise reduction and edge preservation.

Median Filter: It works by moving through the image pixel by pixel, replacing each value with the median value of neighboring pixels. This filter is particularly effective at removing salt-and-pepper noise without blurring edges, which is likely why it shows a more significant improvement in PSNR. It's a good choice for images corrupted by impulse noise.

Gaussian Filter: A Gaussian filter applies a weighted average where the weights decrease with distance from the central pixel, following a Gaussian distribution. It tends to blur edges, which can be detrimental to the preservation of detail, but it's effective at reducing random noise. The improvement in PSNR suggests it's reducing noise, but the loss of edge detail might limit its effectiveness compared to the median filter.

Wiener Filter: This filter aims to deblur an image by taking into account both the degradation function and noise characteristics. It's optimal when the form of the system distortion (PSF) is known accurately, and the noise is stationary and uncorrelated. In practice, however, these conditions are rarely met perfectly, which can lead to suboptimal results, as indicated by the lower PSNR in this case. It's particularly sensitive to the accuracy of the noise estimate and can sometimes enhance noise if the PSF or noise estimate is inaccurate.
"""