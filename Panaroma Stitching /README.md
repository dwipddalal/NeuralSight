
# ImageHomography Class

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
