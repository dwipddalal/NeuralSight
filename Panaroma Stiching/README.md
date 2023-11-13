# Panaroma Stitching

# Code Class Explanation

## FeatureMatcher Class Explanation

The `FeatureMatcher` class is designed to detect and match features between two images. The class makes use of the ORB (Oriented FAST and Rotated BRIEF) keypoint detector and descriptor, along with the brute force matcher for matching these features.

### Initialization (`__init__` method):

The class is initialized with a single parameter:

- **max_features**: This determines the maximum number of features that will be returned after matching. If not specified, it defaults to 30.

Inside the initialization:

- An instance of the ORB detector is created and stored in the `self.orb` attribute.
- A brute-force matcher, specifically designed to work with the Hamming distance (suitable for binary descriptors like ORB), is instantiated and stored in the `self.bf` attribute. The `crossCheck=True` ensures that the matcher only returns consistent pairs.

### Matching Features (`match_features` method):

This method takes in two images and finds the features in each image. Then, it matches these features.

Parameters:

- **img1**: The first input image.
- **img2**: The second input image.

The method works as follows:

1. Detect ORB keypoints and compute the descriptors for both images.
2. Use the brute force matcher to find matches between the descriptors of the two images.
3. Sort the matches based on the distance. This distance represents the similarity of the matched features; smaller distances are better.
4. From the sorted matches, select the top matches as specified by the `max_features` attribute.
5. Extract the coordinates of the matched keypoints.
6. Return the list of matched point pairs, the source points from the first image, and the destination points from the second image.
