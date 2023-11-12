# Histogram Matching

Histogram matching is an image processing technique used to adjust the pixel intensity distribution of an image, so that it matches the pixel intensity distribution of a reference image. 

## Mathematical Overview

Let $A$ be the source image and $B$ be the target image. The histograms for $A$ and $B$ are denoted by $\text{hist}_A$ and $\text{hist}_B$ respectively, and the Cumulative Distribution Functions (CDFs) by $\text{CDF}_A$ and $\text{CDF}_B$.

1. **Normalization of CDFs**: 

$$
\text{CDF}_{A,\text{normalized}} = \frac{\text{CDF}_A - \text{CDF}_A.\text{min}}{\text{CDF}_A.\text{max} - \text{CDF}_A.\text{min}} \times 255
$$
, 
$$
\text{CDF}_{B,\text{normalized}} = \frac{\text{CDF}_B - \text{CDF}_B.\text{min}}{\text{CDF}_B.\text{max} - \text{CDF}_B.\text{min}} \times 255
$$

2. **Mapping Function**:

For each intensity level $i$ in $A$, find the intensity level $j$ in $B$ such that:

$$
j = \arg\min_{k} \left| \text{CDF}_{B,\text{normalized}}[k] - \text{CDF}_{A,\text{normalized}}[i] \right|
$$

3. **Creating the Matched Image**: Replace each pixel $p$ in $A$ with $\text{mapping}[p]$.
