# Non Local Means Denoisinig

### The NLM denoising algorithm is a method used to reduce noise in digital images while preserving important details.


import numpy as np
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numba import jit


def MSE(image1,image2):
  mse=np.mean(np.square(image1 - image2))
  return mse


def PSNR(image1,image2,peak=255):
  mse=MSE(image1,image2)
  psnr=10*np.log10(peak**2/mse)
  return psnr

class ImageProcessor:
    def gaussian_noise(self, img, mean=0, var=0.01):
        img = img / 255
        result = img.copy()
        gauss = np.random.normal(mean, var**0.5, img.shape)
        result = result + gauss
        result = np.clip(result, 0, 1)
        result = np.uint8(result*255)
        return result

    def process_images(self, image_paths):
        noisy_versions = {15: [], 45: [], 80: []}

        for image_path in image_paths:
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            for sigma in [15, 45, 80]:
                noisy_image = self.gaussian_noise(gray_image, mean=0, var=(sigma/255.0)**2)
                noisy_versions[sigma].append(noisy_image)

        return noisy_versions

    def display_images(self, original, noisy_versions):
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(original, cmap='gray')
        plt.title("Original")
        plt.axis('off')

        for i, sigma in enumerate([15, 45, 80], start=2):
            plt.subplot(1, 4, i)
            plt.imshow(noisy_versions[sigma], cmap='gray')
            plt.title(f"Noisy (σ={sigma})")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

image_paths = ['/Users/mihiragarwal/Desktop/Project Courses/Assignment - 1/NLM/image1.png']
noisy_maker = ImageProcessor()
noisy_dataset = noisy_maker.process_images(image_paths)

first_image = cv2.imread(image_paths[0])
first_image_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
first_noisy_versions = {sigma: noisy_dataset[sigma][0] for sigma in [15, 45, 80]}

noisy_maker.display_images(first_image_gray, first_noisy_versions)

def show_gray(img,title=""):
  plt.imshow(img,cmap='gray')
  plt.title(title)


@jit(nopython=True, cache=True)
def find_all_neighbors(pad_img, small_window, big_window, h, w, sigma):
    small_width = small_window // 2
    big_width = big_window // 2
    neighbors = np.zeros((pad_img.shape[0], pad_img.shape[1], small_window, small_window))
    for i in range(big_width, big_width + h):
        for j in range(big_width, big_width + w):
            neighbors[i, j] = pad_img[(i - small_width):(i + small_width + 1), (j - small_width):(j + small_width + 1)]
    return neighbors

@jit(nopython=True, cache=True)
def evaluate_norm(pixel_window, neighbor_window, h, sigma):
    ip_numerator, Z = 0, 0
    f = pixel_window.shape[0] // 2  # small_width in your original code
    normalization_factor = 1 / ((2 * f + 1) ** 2)

    for i in range(neighbor_window.shape[0]):
        for j in range(neighbor_window.shape[1]):
            q_window = neighbor_window[i, j]
            d2 = np.sum((pixel_window - q_window)**2) * normalization_factor
            w = np.exp(-max(d2 - 2 * sigma * 2, 0) / h * 2)
            q_x, q_y = q_window.shape[0] // 2, q_window.shape[1] // 2
            Iq = q_window[q_x, q_y]
            ip_numerator += w * Iq
            Z += w

    return ip_numerator / Z

class NLMeans():

  def solve(self, img, h=30, small_window=7, big_window=21, sigma=0.1):
    pad_img = np.pad(img, big_window // 2, mode='reflect')

    return self.nlm(pad_img, img, h, small_window, big_window, sigma)

  @staticmethod
  @jit(nopython=True, cache=True)
  def nlm(pad_img, img, h, small_window, big_window, sigma):
    Nw = (h ** 2) * (small_window ** 2)
    h, w = img.shape
    result = np.zeros(img.shape)
    big_width = big_window // 2
    small_width = small_window // 2
    neighbors = find_all_neighbors(pad_img, small_window, big_window, h, w, sigma)
    for i in range(big_width, big_width + h):
        for j in range(big_width, big_width + w):
            pixel_window = neighbors[i, j]
            neighbor_window = neighbors[(i - big_width):(i + big_width + 1), (j - big_width):(j + big_width + 1)]
            ip = evaluate_norm(pixel_window, neighbor_window, Nw, sigma)
            result[i - big_width, j - big_width] = max(min(255, ip), 0)

    return result

def process_and_display(image_path, denoiser, noise_maker):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    k_values = [0.1]
    f_values = [1, 2, 3, 4]  # Corresponding to window sizes {3, 5, 7, 9}

    for sigma in [80, 45, 15]:
        for k in k_values:
            h = k * sigma
            for f in f_values:
                small_window = 2*f + 1
                if sigma == 80:
                    big_window = 35
                else:
                    big_window = 21

                gaussian_example = noise_maker.gaussian_noise(image.copy(), var=(sigma/255.0)**2)
                custom_gaussian_denoise = denoiser.solve(gaussian_example.copy(), h=h, small_window=small_window, big_window=big_window, sigma=sigma)
                psnr_value = PSNR(custom_gaussian_denoise, image)
                mse_value = MSE(custom_gaussian_denoise, image)
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                show_gray(gaussian_example, f"Noisy (σ={sigma})")
                plt.subplot(1, 2, 2)
                show_gray(custom_gaussian_denoise, f"Denoised (k={k:.2f}, Window={small_window},  => PSNR: {psnr_value:.2f}, MSE: {mse_value:.2f})")
                plt.tight_layout()
                plt.show()

for i in [2,3,4,1]:
    denoiser = NLMeans()
    noise_maker = ImageProcessor()
    image_path = f'image_path/NLM/image{i}.png'
    process_and_display(image_path, denoiser, noise_maker)

