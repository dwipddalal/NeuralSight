# Histogram Matching

### Importing Libraries
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.color import rgb2gray, gray2rgb
from skimage.exposure import match_histograms

"""### Reading Images"""

img1 = cv.imread("/Users/mihiragarwal/Desktop/Project Courses/Assignment - 1/histogram_matching/Source.png", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("/Users/mihiragarwal/Desktop/Project Courses/Assignment - 1/histogram_matching/Target.png", cv.IMREAD_GRAYSCALE)

plt.figure(figsize=(12, 5))

# Plot img1 on the left subplot
plt.subplot(1, 2, 1)
plt.title('Source image')
plt.imshow(img1, cmap=cm.gray)
plt.axis('off')

# Plot img2 on the right subplot
plt.subplot(1, 2, 2)
plt.title('Target Image')
plt.imshow(img2, cmap=cm.gray)
plt.axis('off')

plt.show()

"""### Function to calculate CDF and histogram"""

def calculate_histogram_and_cdf(image):
    histogram = np.zeros(256, dtype=int)

    for pixel_value in image.flatten():
        histogram[pixel_value] += 1

    cdf = histogram.cumsum()

    return histogram, cdf

hist1, cdf1 = calculate_histogram_and_cdf(img1)
hist2, cdf2 = calculate_histogram_and_cdf(img2)

plt.figure(figsize=(12, 6))

# Plot hist1 on the left subplot
plt.subplot(1, 2, 1)
plt.bar(range(256), hist1, width=1.0, color='b')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Source Image')

# Plot hist2 on the right subplot
plt.subplot(1, 2, 2)
plt.bar(range(256), hist2, width=1.0, color='r')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Target Image')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

# Plot cdf1 on the left subplot
plt.subplot(1, 2, 1)
plt.bar(range(256), cdf1, width=1.0, color='b')
plt.xlabel('Pixel')
plt.ylabel('CDF')
plt.title('CDF of Source Image')

# Plot cdf2 on the right subplot
plt.subplot(1, 2, 2)
plt.bar(range(256), cdf2, width=1.0, color='r')
plt.xlabel('Pixel')
plt.ylabel('CDF')
plt.title('CDF of Target Image')

plt.tight_layout()
plt.show()

"""### Normalizing the CDF"""

cdf_a_normalized = (cdf1 - cdf1.min()) * 255 / (cdf1.max() - cdf1.min())
cdf_b_normalized = (cdf2 - cdf2.min()) * 255 / (cdf2.max() - cdf2.min())

plt.figure(figsize=(12, 6))

# Plot cdf_a_normalized on the left subplot
plt.subplot(1, 2, 1)
plt.bar(range(256), cdf_a_normalized, width=1.0, color='b')
plt.xlabel('Pixel')
plt.ylabel('CDF Normalized')
plt.title('CDF Normalized of Source Image')

# Plot cdf_b_normalized on the right subplot
plt.subplot(1, 2, 2)
plt.bar(range(256), cdf_b_normalized, width=1.0, color='r')
plt.xlabel('Pixel')
plt.ylabel('CDF Normalized')
plt.title('CDF Normalized of Target Image')

plt.tight_layout()
plt.show()

"""### Mapping the input image to the output image"""

mapping = np.zeros(256, dtype=int)
for i in range(256):
    mapping[i] = np.argmin(np.abs(cdf_b_normalized - cdf_a_normalized[i]))

matched_image = mapping[img1]

img1_rgb = gray2rgb(img1)
img2_rgb  = gray2rgb(img2)
matched = match_histograms(img1_rgb, img2_rgb, channel_axis=-1)

plt.figure(figsize=(18, 6))

# Plot img1 on the left subplot
plt.subplot(1, 4, 1)
plt.title('Source image')
plt.imshow(img1, cmap=cm.gray)
plt.axis('off')

# Plot img2 in the middle subplot
plt.subplot(1, 4, 2)
plt.title('Target Image')
plt.imshow(img2, cmap=cm.gray)
plt.axis('off')

# Plot matched_image on the right subplot
plt.subplot(1, 4, 3)
plt.title('Matched Image')
plt.imshow(matched_image, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Skimage Matched Image')
plt.imshow(matched, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.tight_layout()
plt.show()


def histogram_matching(img1, img2):
    hist1, cdf1 = calculate_histogram_and_cdf(img1)
    hist2, cdf2 = calculate_histogram_and_cdf(img2)
    cdf_a_normalized = (cdf1 - cdf1.min()) * 255 / (cdf1.max() - cdf1.min())
    cdf_b_normalized = (cdf2 - cdf2.min()) * 255 / (cdf2.max() - cdf2.min())
    mapping = np.zeros(256, dtype=int)
    for i in range(256):
        mapping[i] = np.argmin(np.abs(cdf_b_normalized - cdf_a_normalized[i]))

    matched_image = mapping[img1]
    img1_rgb = gray2rgb(img1)
    img2_rgb  = gray2rgb(img2)
    matched = match_histograms(img1_rgb, img2_rgb, channel_axis=-1)

    plt.figure(figsize=(18, 6))

    # Plot img1 on the left subplot
    plt.subplot(1, 4, 1)
    plt.title('Source image')
    plt.imshow(img1, cmap=cm.gray)
    plt.axis('off')

    # Plot img2 in the middle subplot
    plt.subplot(1, 4, 2)
    plt.title('Target Image')
    plt.imshow(img2, cmap=cm.gray)
    plt.axis('off')

    # Plot matched_image on the right subplot
    plt.subplot(1, 4, 3)
    plt.title('Matched Image')
    plt.imshow(matched_image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title('Skimage Matched Image')
    plt.imshow(matched, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

for i in range (1,5):
    for j in range(1,5):
        if ((i==1 and j==2) or (i==4 and j==3) or (i==2 and j==4) or (i==3 and j==1)):
            img1 = cv.imread(f"/Users/mihiragarwal/Desktop/Project Courses/Assignment - 1/histogram_matching/grey_{i}.png", cv.IMREAD_GRAYSCALE)
            img2 = cv.imread(f"/Users/mihiragarwal/Desktop/Project Courses/Assignment - 1/histogram_matching/grey_{j}.png", cv.IMREAD_GRAYSCALE)
            histogram_matching(img1, img2)

"""# Observations

- As a result of the pixel mapping, the visual appearance of image A will change. If the histograms of A and B were initially different, this mapping will bring image A closer in appearance to image B in terms of pixel intensity distribution.

- The Skimage library provides a function called `match_histograms` that implements the above steps. The function takes in two images and returns a matched image. My custom implementation and the Skimage implementation are compared. Most of the time they give similar images, but huge deviations are also observed.

"""