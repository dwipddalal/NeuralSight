# Histogram Matching

Histogram matching is an image processing technique used to adjust the pixel intensity distribution of an image, so that it matches the pixel intensity distribution of a reference image. 

## Mathematical Overview

Let $A$ be the source image and $B$ be the target image. The histograms for $A$ and $B$ are denoted by $\text{hist}_A$ and $\text{hist}_B$ respectively, and the Cumulative Distribution Functions (CDFs) by $\text{CDF}_A$ and $\text{CDF}_B$.

1. **Normalization of CDFs**: 


<img width="525" alt="image" src="https://github.com/dwipddalal/NeuralSight/assets/91228207/428b9f4d-f0bc-41b5-8beb-c4f13a0a7a42">


2. **Mapping Function**:

For each intensity level $i$ in $A$, find the intensity level $j$ in $B$ such that:

<img width="496" alt="image" src="https://github.com/dwipddalal/NeuralSight/assets/91228207/fb33742a-0622-4cb7-9f88-640a8262eac3">


3. **Creating the Matched Image**: Replace each pixel $p$ in $A$ with $\text{mapping}[p]$.

## Results:
<img width="877" alt="image" src="https://github.com/dwipddalal/NeuralSight/assets/91228207/e13159ad-a668-4dae-84bc-c42e5a5d82d4">
<img width="878" alt="image" src="https://github.com/dwipddalal/NeuralSight/assets/91228207/433ae5f2-fcf4-4f34-8ccf-335cc650afbd">
<img width="877" alt="image" src="https://github.com/dwipddalal/NeuralSight/assets/91228207/83ec9184-7993-4685-ac8f-9b93df389375">
<img width="875" alt="image" src="https://github.com/dwipddalal/NeuralSight/assets/91228207/da67c143-df43-4e19-bad4-9bb09238dcc5">




## Observations:

- As a result of the pixel mapping, the visual appearance of image A will change. If the histograms of A and B were initially different, this mapping will bring image A closer in appearance to image B in terms of pixel intensity distribution.

- The Skimage library provides a function called `match_histograms` that implements the above steps. The function takes in two images and returns a matched image. My custom implementation and the Skimage implementation are compared. Most of the time they give similar images, but huge deviations are also observed.
