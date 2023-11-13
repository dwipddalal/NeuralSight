from blending import Blender
import cv2
import glob
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class FeatureMatcher:
    def __init__(self, max_features=50):
        """
        Initializes the FeatureMatcher with the desired number of features.

        Parameters:
        - max_features: Maximum number of features to return.
        """
        self.max_features = max_features
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match_features(self, img1, img2):
        """
        Detects and matches ORB features between two images.

        Parameters:
        - img1: First input image.
        - img2: Second input image.

        Returns:
        - matched_points: List of matched point pairs.
        - src_points: Source points from the first image.
        - dst_points: Destination points from the second image.
        """
        keypoints1, descriptors1 = self.orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(img2, None)

        matches = self.bf.match(descriptors1, descriptors2)
        matches_sorted = sorted(matches, key=lambda x: x.distance)

        matched_points = [(keypoints1[m.queryIdx].pt, keypoints2[m.trainIdx].pt) for m in matches_sorted[:self.max_features]]
        print(f'Total of {len(matched_points)} matches found.')

        src_points = np.float32([point[0] for point in matched_points]).reshape(-1, 1, 2)
        dst_points = np.float32([point[1] for point in matched_points]).reshape(-1, 1, 2)

        return np.array(matched_points), src_points, dst_points

    def plot_matches(self, img1, img2, matched_points):
        """
        Plots the matched keypoints between two images.
        """
        keypoints1, _ = self.orb.detectAndCompute(img1, None)
        keypoints2, _ = self.orb.detectAndCompute(img2, None)
        matches_sorted = sorted(self.bf.match(_, _), key=lambda x: x.distance)[:self.max_features]

        img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches_sorted, None)

        plt.figure(figsize=(12, 6))

        plt.imshow(img_matches)
        plt.title('Matched Keypoints')
        plt.show()

# Usage
matcher = FeatureMatcher()

img1 = cv2.imread('Dataset/scene1/I11.JPG')
img2 = cv2.imread('Dataset/scene1/I12.JPG')

matched_points, _, _ = matcher.match_features(img1, img2)
matcher.plot_matches(img1, img2, matched_points)

"""# ImageHomography Class

The `ImageHomography` class is designed to handle operations related to computing and applying image homographies.

## Overview

An image homography is a 3x3 matrix that describes the projective transformation between two 2D images, usually used for image stitching or perspective warping.

## Methods

### `compute_homography(matches)`

#### Purpose:
Computes a homography matrix given a set of point matches between two images.

#### Parameters:
- `matches`: A list of matched points between two images. Each match is a tuple of two points, where each point is a tuple `(x, y)`.

#### How it works:
It constructs a matrix `A` with 2 rows for each match and uses Singular Value Decomposition (SVD) to solve for the homography matrix `H`.

**Math Involved**:
The equations are derived from the point correspondences using the definition of a homography transformation.

### `compute_best_homography_ransac(correspondences, trials=10000, distance_threshold=10, num_samples=4)`

#### Purpose:
Computes the best homography matrix using the RANSAC (RANdom SAmple Consensus) algorithm.

#### Parameters:
- `correspondences`: List of point correspondences between two images.
- `trials`: Number of RANSAC trials.
- `distance_threshold`: The threshold distance for inlier determination.
- `num_samples`: Number of random samples to pick for computing homography in each trial.

#### How it works:
The method randomly selects a subset of point correspondences, computes a homography using these points, and determines inliers based on the `distance_threshold`. The homography that has the most inliers across all trials is considered the best.

**Math Involved**:
RANSAC is a non-deterministic algorithm that iteratively estimates the parameters of a model. The logic behind RANSAC is that inliers follow the model's hypothesis while outliers do not.

### `transform_point(i, j, H)`

#### Purpose:
Transforms a point `(i, j)` using a given homography matrix `H`.

#### Parameters:
- `i`, `j`: Coordinates of the point.
- `H`: Homography matrix.

#### How it works:
It applies the homography transformation to the point and normalizes the result by the third coordinate.

### `transform_image(source_img, homography_matrix, target_img, offset=[0, 0])`

#### Purpose:
Transforms the `source_img` using the `homography_matrix` and places the transformed image on the `target_img`.

#### Parameters:
- `source_img`: The image to be transformed.
- `homography_matrix`: The transformation matrix.
- `target_img`: The target/base image where the transformed image will be placed.
- `offset`: Offset for placing the transformed image on the target.

#### How it works:
1. Computes the inverse of the homography matrix.
2. Transforms the four corners of the `source_img` to determine the bounding box of the transformed image in the target's coordinate space.
3. For each pixel in this bounding box, the method computes its corresponding pixel in the `source_img` using the inverse transformation.
4. The pixel values are then copied from the `source_img` to the `target_img`.

**Math Involved**:
Inverse transformation is used to ensure that all pixels in the transformed image get filled and to prevent aliasing.

---

The `ImageHomography` class provides a comprehensive set of methods for computing and applying image homographies. The incorporation of the RANSAC algorithm helps in robustly estimating the homography even when there are mismatched point correspondences. The transform methods allow for the warping of images and overlaying them on a target image, which is especially useful for applications like image stitching.
"""

class ImageHomography:
    def __init__(self):
        pass

    @staticmethod
    def compute_homography(matches):
        A = np.zeros((2*len(matches), 9))
        for i, ((x1, y1), (x2, y2)) in enumerate(matches):
            A[2*i] = [x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2]
            A[2*i+1] = [0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2]

        _, _, V = np.linalg.svd(A)
        H = np.reshape(V[-1], (3, 3))
        return H

    def compute_best_homography_ransac(self, correspondences, trials=10000, distance_threshold=10, num_samples=4):
        '''
        Applies the RANSAC algorithm to compute the best homography matrix using random sampling.
        '''
        best_homography = None
        max_num_inliers = 0
        selected_sample = None

        for trial in tqdm(range(trials)):
            inliers = []

            # Randomly sample correspondences and compute the homography matrix.
            # If the number of inliers is the best so far, update the best homography.
            selected_sample = correspondences[np.random.choice(len(correspondences), size=num_samples, replace=False)]
            homography_matrix = self.compute_homography(selected_sample)

            for correspondence in correspondences:
                source_point = np.append(correspondence[0], 1).T
                target_point = np.append(correspondence[1], 1).T
                transformed_point = np.dot(homography_matrix, source_point)
                transformed_point /= transformed_point[2]

                if np.linalg.norm(transformed_point - target_point) < distance_threshold:
                    inliers.append(correspondence)

            # If this homography gives maximum inliers so far, update.
            if len(inliers) > max_num_inliers:
                max_num_inliers = len(inliers)
                best_homography = homography_matrix

        print('Max number of inliers = ', max_num_inliers)
        return best_homography, selected_sample

    @staticmethod
    def transform_point(i, j, H):
        transformed = np.dot(H, [i, j, 1])
        return (transformed[:2] / transformed[2]).astype(np.int)


    def transform_image(self, source_img, homography_matrix, target_img, offset=[0, 0]):
        """Transforms the source image using the homography matrix and places it on the target image."""

        height, width, _ = source_img.shape

        # Compute inverse homography matrix to prevent aliasing and holes in the output image.
        inverse_homography = np.linalg.inv(homography_matrix)

        # Corners of the source image
        top_left = self.transform_point(0, 0, homography_matrix)
        top_right = self.transform_point(width - 1, 0, homography_matrix)
        bottom_left = self.transform_point(0, height - 1, homography_matrix)
        bottom_right = self.transform_point(width - 1, height - 1, homography_matrix)

        bounding_box = np.array([top_left, top_right, bottom_left, bottom_right])
        min_x = np.min(bounding_box[:, 0])
        max_x = np.max(bounding_box[:, 0])
        min_y = np.min(bounding_box[:, 1])
        max_y = np.max(bounding_box[:, 1])

        # Generate coordinates for the transformed bounding box
        coordinates = np.indices((max_x - min_x, max_y - min_y)).reshape(2, -1)
        coordinates = np.vstack((coordinates, np.ones(coordinates.shape[1]))).astype(np.int)

        coordinates[0, :] += min_x
        coordinates[1, :] += min_y

        # Compute the pixel positions in the source image using the inverse transformation
        transformed_coords = np.dot(inverse_homography, coordinates)
        target_y, target_x = coordinates[1, :], coordinates[0, :]

        # Convert to cartesian coordinates
        source_y = (transformed_coords[1, :] / transformed_coords[2, :]).astype(np.int)
        source_x = (transformed_coords[0, :] / transformed_coords[2, :]).astype(np.int)

        # Ensure that coordinates are within image boundaries
        valid_indices = np.where((source_y >= 0) & (source_y < height) & (source_x >= 0) & (source_x < width))

        source_x = source_x[valid_indices]
        source_y = source_y[valid_indices]
        target_x = target_x[valid_indices]
        target_y = target_y[valid_indices]

        # Assign pixel values to the target image
        target_img[target_y + offset[1], target_x + offset[0]] = source_img[source_y, source_x]

"""## ImageBlender Class Explanation

This class is dedicated to blending two images smoothly by using the concept of image pyramids, particularly Gaussian and Laplacian pyramids.

### Initialization: `__init__(self, pyramid_depth=6)`

This method initializes the `ImageBlender` class with a specific pyramid depth. The depth of the pyramid determines the number of layers in the Gaussian or Laplacian pyramid.

### Gaussian Pyramid: `compute_gaussian_pyramid(self, image)`

A Gaussian pyramid is constructed by repeatedly reducing the size of an image. At each level, the image is smoothed (typically with a Gaussian filter) and then subsampled. In the provided method, the Gaussian pyramid is computed by successively downsampling the image using the `cv2.pyrDown()` function.

**Math Involved**: Gaussian blurring is applied before down-sampling, which involves convolving the image with a Gaussian filter.

### Laplacian Pyramid: `compute_laplacian_pyramid(self, image)`

The Laplacian pyramid is constructed from the Gaussian pyramid. For each level of the Gaussian pyramid, we subtract the next level upsampled. This gives us a band-pass image representation. The method computes this by downsampling the image and then upsampling the result to subtract from the original. The difference gives the Laplacian for that level.

**Math Involved**: The subtraction of two successive Gaussian blurred images to get the band-pass representation.

### Pyramid Blending: `blend_pyramids(self, laplacian_a, laplacian_b, gaussian_mask)`

Given two Laplacian pyramids (from two images) and a Gaussian pyramid (of a mask), the two Laplacian pyramids are blended. This is done by taking a weighted sum of their values based on the Gaussian mask.

**Math Involved**: Linear interpolation of two images based on a mask.

### Reconstruct Image: `reconstruct_image(self, laplacian_pyramid)`

Reconstructs an image from its Laplacian pyramid. Starting from the smallest image in the pyramid, we progressively upsample and add the corresponding layer from the Laplacian pyramid until the original image size is reached.

### Extract Image Mask: `extract_image_mask(self, image)`

Extracts a binary mask from the image, where the mask indicates which parts of the image are non-zero.

### Image Blending: `blend_images(self, image1, image2, strategy='STRAIGHTCUT')`

The core method for blending two images using specified blending strategies:

1. **STRAIGHTCUT**: This strategy takes a simple average of the two images where they overlap and takes complete values from one image on the left half and the other image on the right half.

2. **DIAGONAL**: In this method, a convex polygon is defined to dictate the blending region.

The blending process involves:

- Calculating Laplacian pyramids for both images.
- Extracting masks for both images to determine overlapping regions.
- Creating a blending mask based on the specified strategy.
- Computing the Gaussian pyramid for the blending mask.
- Using the blending mask pyramid to blend the two Laplacian pyramids.
- Reconstructing the blended image from the blended Laplacian pyramid.

**Math Involved**:

- **Convex Polygon Masking**: In the DIAGONAL strategy, a convex polygon is used to define the blending region. This involves geometric computations to determine pixels that lie inside the polygon.

"""

class ImageBlender():

    def __init__(self, pyramid_depth=6):
        """Initializes the ImageBlender class."""
        self.pyramid_depth = pyramid_depth

    def compute_gaussian_pyramid(self, image):
        """
        Compute the Gaussian pyramid for an image.
        """
        pyramid = [image]
        for _ in range(self.pyramid_depth - 1):
            downscaled = cv2.pyrDown(pyramid[-1])
            pyramid.append(downscaled)
        return pyramid

    def compute_laplacian_pyramid(self, image):
        """
        Compute the Laplacian pyramid for an image.
        """
        pyramid = []
        for _ in range(self.pyramid_depth-1):
            downscaled_image = cv2.pyrDown(image)
            size = (image.shape[1], image.shape[0])
            upscaled_downscaled = cv2.pyrUp(downscaled_image, dstsize=size)
            difference =  image.astype(float) - upscaled_downscaled.astype(float)
            pyramid.append(difference)
            image = downscaled_image

        pyramid.append(image)
        return pyramid

    def blend_pyramids(self, laplacian_a, laplacian_b, gaussian_mask):
        """
        Blends two Laplacian pyramids using a Gaussian pyramid mask.
        """
        blended_pyramid = []
        for i, mask in enumerate(gaussian_mask):
            merged_mask = cv2.merge((mask, mask, mask))
            blended_image = laplacian_a[i]*merged_mask + laplacian_b[i]*(1 - merged_mask)
            blended_pyramid.append(blended_image)

        return blended_pyramid

    def reconstruct_image(self, laplacian_pyramid):
        """
        Reconstructs the image from a Laplacian pyramid.
        """
        reconstructed_image = laplacian_pyramid[-1]
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            laplacian_level = laplacian_pyramid[i]
            size = laplacian_level.shape[:2][::-1]
            reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=size).astype(float)
            reconstructed_image += laplacian_level.astype(float)

        return reconstructed_image

    def extract_image_mask(self, image):
        """
        Extracts the binary mask of an image.
        """
        mask = image[:, :, 0] != 0
        mask = np.logical_and(image[:, :, 1] != 0, mask)
        mask = np.logical_and(image[:, :, 2] != 0, mask)

        mask_image = np.zeros(image.shape[:2], dtype=float)
        mask_image[mask] = 1.0
        return mask_image, mask

    def blend_images(self, image1, image2, strategy='STRAIGHTCUT'):
        """
        Blend two images using specified blending strategy.
        """
        laplacian_1 = self.compute_laplacian_pyramid(image1)
        laplacian_2 = self.compute_laplacian_pyramid(image2)

        _, mask1_binary = self.extract_image_mask(image1)
        _, mask2_binary = self.extract_image_mask(image2)

        overlap = mask1_binary & mask2_binary
        y_indices, x_indices = np.where(overlap)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        height, _ = overlap.shape

        blending_mask = np.zeros(image1.shape[:2])
        if strategy == 'STRAIGHTCUT':
            blending_mask[:, :(x_min + x_max)//2] = 1.0
        elif strategy == 'DIAGONAL':
            blending_mask = cv2.fillConvexPoly(blending_mask, np.array([
                [[x_min, y_min], [x_max, y_max], [x_max, height], [0, height], [0, 0], [x_min, 0]]
            ]), True, 50)

        gaussian_mask = self.compute_gaussian_pyramid(blending_mask)
        blended_pyramid = self.blend_pyramids(laplacian_1, laplacian_2, gaussian_mask)
        final_image = self.reconstruct_image(blended_pyramid)
        return final_image, mask1_binary, mask2_binary

"""## **execute_stitching Function**
`execute_stitching` performs image stitching on a given pair of images using either a custom homography method or OpenCV's built-in method. Let's break down the function:

### **Parameters**:

- `index1`, `index2`: Indices of the two images to be stitched together.
- `prevH`: Previous homography matrix.
- `image_paths`: A list containing paths to the images.
- `dataset`: Identifier for the dataset being used.
- `shape`: The shape to resize the images to, default is `(600, 400)`.
- `offset`: Offset to position the transformed image, default is `[1200, 800]`.
- `trials`: Number of RANSAC trials for custom method, default is `5000`.
- `threshold`: Distance threshold for RANSAC, default is `2`.
- `method`: Specifies the method to use for homography calculation; can be `'Custom'` or `'OpenCV'`.

### **Functionality**:

1. **Initialize**:
    - Create an empty image `warped_image` with a size of `(2500, 3500, 3)`.
    - Read the two images using OpenCV based on the provided indices.
    - Resize the images to the specified shape.

2. **Feature Matching**:
    - Use a `FeatureMatcher` (not provided in the code snippet) to extract and match features between `img2` and `img1`.
    - The matched features are returned as `matches`, with source points `src` from `img2` and destination points `dst` from `img1`.

3. **Homography Calculation**:
    - If the `method` is `'Custom'`:
        - An instance of `ImageHomography` is created.
        - The best homography matrix `H` is computed using the custom method with RANSAC.
        - The current `H` is combined with `prevH` by matrix multiplication to get the accumulated homography.
        - The second image `img2` is transformed using the accumulated homography and placed on the `warped_image` with the specified offset.
        - The resulting stitched image is saved to the disk.
    - If the `method` is `'OpenCV'`:
        - Use OpenCV's `findHomography` function to compute the homography matrix.
        - Similar to the custom method, the current `H` is combined with `prevH`.
        - OpenCV's `warpPerspective` function is used to warp `img2` using the accumulated homography.
        - The resulting stitched image is saved to the disk.

### **Return Value**:

- The function returns the accumulated homography matrix `prevH`.

"""

def execute_stitching(index1, index2, prevH, image_paths, dataset, shape = (600, 400), offset = [1200, 800], trials=5000, threshold=2, method = 'Custom'):
    '''
    Function that, for a given pair of indices, computes the best homography and saves the warped images to disk.
    '''
    warped_image = np.zeros((2500, 3500, 3))
    img1 = cv2.imread(image_paths[index1])
    img2 = cv2.imread(image_paths[index2])
    img1 = cv2.resize(img1, shape)
    img2 = cv2.resize(img2, shape)
    feature = FeatureMatcher()
    matches, src, dst = feature.match_features(img2, img1)
    if method == 'Custom':
        homography_calculator = ImageHomography()
        H, _ = homography_calculator.compute_best_homography_ransac(matches, trials = trials, distance_threshold = threshold)
        prevH = np.dot(prevH, H)
        homography_calculator.transform_image(img2, prevH, target_img = warped_image, offset = offset)
        cv2.imwrite('Output/scene' + str(dataset) + '/Custom/Warp_' + str(index2) +  '.png', warped_image)
        return prevH
    elif method == 'OpenCV':
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, threshold)
        prevH = np.dot(prevH, H)
        warped_image = cv2.warpPerspective(img2, prevH, (warped_image.shape[1], warped_image.shape[0]))
        cv2.imwrite('Output/scene' + str(dataset) + '/OpenCV/Warp_' + str(index2) + '.png', warped_image)
        return prevH

"""## Main method

#### Custom
"""

dataset = 2
image_paths = sorted(glob.glob('Dataset/scene' + str(dataset) + '/*jpg'))
print(len(image_paths))
os.makedirs('Output/scene' + str(dataset) + '/Custom/', exist_ok = True)
os.makedirs('Output/scene' + str(dataset) + '/OpenCV/', exist_ok = True)
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
shape = (600, 400)
mid = len(image_paths)//2
threshold = 2
trials = 5000
offset = [1200, 800]
prevH = np.eye(3)
if dataset < 4:
    prevH = execute_stitching(2, 1, prevH, image_paths =  image_paths, dataset =  dataset, method= 'Custom')
    prevH = execute_stitching(1, 0, prevH, image_paths =  image_paths, dataset =  dataset, method= 'Custom')
    prevH = np.eye(3)
    prevH = execute_stitching(2, 2, prevH, image_paths =  image_paths, dataset =  dataset, method= 'Custom')
    prevH = np.eye(3)
    prevH = execute_stitching(2, 3, prevH, image_paths =  image_paths, dataset =  dataset, method= 'Custom')
else:
    prevH = execute_stitching(1, 0, prevH, image_paths =  image_paths, dataset =  dataset, method= 'Custom')
    prevH = np.eye(3)
    prevH = execute_stitching(1, 1, prevH, image_paths =  image_paths, dataset =  dataset, method= 'Custom')

b = ImageBlender()
final_image =  cv2.imread('Output/scene' + str(dataset) + '/Custom/'  + 'Warp_' + str(0) + '.png')
length = 4 if dataset < 4 else 2
for index in range(0, length):
    print('blending', index)
    img2 = cv2.imread('Output/scene' + str(dataset) + '/Custom/' + 'Warp_' + str(index) + '.png')
    final_image, mask_t, mask_2t = b.blend_images(final_image, img2)
    mask_t = mask_t + mask_2t
    cv2.imwrite('Output/scene' + str(dataset) + '/Custom/' 'FINAL.png', final_image)

"""#### Build-in"""

offset = [1500, 800]
offsetMatrix = np.array([[1, 0, offset[0]],
                    [0, 1, offset[1]],
                    [0, 0, 1]])
prevH = offsetMatrix.copy()
if dataset < 4:
    prevH = execute_stitching(2, 1, prevH, image_paths =  image_paths, dataset =  dataset, method= 'OpenCV')
    prevH = execute_stitching(1, 0, prevH, image_paths =  image_paths, dataset =  dataset, method= 'OpenCV')
    prevH = offsetMatrix.copy()
    prevH = execute_stitching(2, 2, prevH, image_paths =  image_paths, dataset =  dataset, method= 'OpenCV')
    prevH = offsetMatrix.copy()
    prevH = execute_stitching(2, 3, prevH, image_paths =  image_paths, dataset =  dataset, method= 'OpenCV')
else:
    prevH = execute_stitching(1, 0, prevH, image_paths =  image_paths, dataset =  dataset, method= 'OpenCV')
    prevH = offsetMatrix.copy()
    prevH = execute_stitching(1, 1, prevH, image_paths =  image_paths, dataset =  dataset, method= 'OpenCV')
b = ImageBlender()
finalImg =  cv2.imread('Output/scene' + str(dataset) + '/OpenCV/'  + 'Warp_' + str(0) + '.png')
length = 4 if dataset < 4 else 2

for index in range(0, length):
    print('blending', index)
    img2 = cv2.imread('Output/scene' + str(dataset) + '/OpenCV/' + 'Warp_' + str(index) + '.png')
    finalImg, mask_t, mask_2t = b.blend_images(finalImg, img2)
    mask_t = mask_t + mask_2t
    cv2.imwrite('Output/scene' + str(dataset) + '/OpenCV/' 'FINAL.png', finalImg)

"""## Dataset Output
### Scene 2 Custom:
![image.png](attachment:image.png)

### Scene 2 Open CV:
![image-2.png](attachment:image-2.png)

---

### Scene 4 Custom:
![image-3.png](attachment:image-3.png)

### Scene 4 Open CV
![image-4.png](attachment:image-4.png)


---

### Scene 5 Custom
![image.png](attachment:image.png)

### Scene 5 Open CV
![image-2.png](attachment:image-2.png)

## Citations

- Use of chatgpt for comments and descriptive variables
"""
