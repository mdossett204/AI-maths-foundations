# Understanding Convolutional Neural Networks (CNNs)

## 1. Image Representation

Images are represented as tensors of pixel intensities:

- **Grayscale:** A 2D matrix (Height x Width).
- **RGB:** A 3D tensor (Height x Width x 3 Channels).
- **Values:** Typically range from **0 to 255**, though they are often normalized to [0, 1] for training.

## 2. The Convolution Operation

A kernel (filter) slides across the input image, performing element-wise multiplication and summing the results to create a feature map.

### Mathematical Example

Input (3x3):

```
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
```

Kernel (2x2):

```
[[1, 0],
 [0, 1]]
```

**Calculation:**

1. Top-Left: `(1*1) + (2*0) + (4*0) + (5*1) = 6`
2. Top-Right: `(2*1) + (3*0) + (5*0) + (6*1) = 8`
3. Bottom-Left: `(4*1) + (5*0) + (7*0) + (8*1) = 12`
4. Bottom-Right: `(5*1) + (6*0) + (8*0) + (9*1) = 14`

**Resulting Feature Map:**

```
[[6, 8],
 [12, 14]]
```

## 3. Handling RGB Channels

When processing RGB images:

1.  **Filter Depth:** Each filter has a depth matching the input (e.g., a 3x3 filter is actually 3x3x3).
2.  **Summation:** The convolution is performed on all channels simultaneously, and the results are summed into a **single** 2D feature map per filter.
3.  **Multiple Filters:** To get a 3D output (like 32 channels), you apply 32 distinct filters.

## 4. Output Dimension Formula

To calculate the output size ($O$) given input size ($I$), kernel size ($K$), padding ($P$), and stride ($S$):

$$O = \left\lfloor \frac{I + 2P - K}{S} \right\rfloor + 1$$

## 5. Summary of Terms

- **Stride ($S$):** How many pixels the filter shifts at each step.
- **Padding ($P$):** Adding extra pixels (usually zeros) around the border to control output size.
- **ReLU:** An activation function applied after convolution to introduce non-linearity.
- **Feature Map:** The output of a convolution layer representing detected patterns.

## 6. Multiple Filters and Output Depth

In practice, a single convolutional layer uses many filters (e.g., 32, 64, or 128) to learn different features from the same input.

- **Feature Diversity:** Each filter has its own set of weights. One might learn to detect vertical edges, while another detects color gradients or specific textures.
- **Output Volume:** If you apply $N$ filters to an input, the output will have a depth of $N$. For example, if your input is $224 \times 224 \times 3$ and you apply 64 filters, your output feature map will have a depth of 64 ($O_h \times O_w \times 64$).
- **Parameter Calculation:** Each filter must match the depth of the input. If the input has $D_{in}$ channels and the filter size is $K \times K$, then:
  - Weights per filter: $K \times K \times D_{in}$
  - Total parameters in layer: $N \times (K \times K \times D_{in} + 1)$ (the $+1$ is for the bias term per filter).

## 7. Detailed RGB Example with Bias

### The Bias Term ($b$)

The bias is a trainable constant added to the output of each filter. It acts as an offset, allowing the model to better fit the data by shifting the activation function. There is **one bias per filter**.

### Walkthrough: 3x3x3 Input with 2 Filters (2x2x3)

**Input (RGB):**

- $I_R$: `[[1,0,1],[0,1,0],[1,0,1]]`
- $I_G$: `[[0,1,0],[1,0,1],[0,1,0]]`
- $I_B$: `[[1,1,1],[1,1,1],[1,1,1]]`

**Filter 1 ($W_1$) Weights & Bias ($b_1=1$):**

- $W_{1R}: [[1,0],[0,1]]$, $W_{1G}: [[0,1],[1,0]]$, $W_{1B}: [[1,1],[1,1]]$

**Calculation for Top-Left Output ($O_{1,1}$):**

1.  **Convolve Red:** $(1\times1 + 0\times0 + 0\times0 + 1\times1) = 2$
2.  **Convolve Green:** $(0\times0 + 1\times1 + 1\times1 + 0\times0) = 2$
3.  **Convolve Blue:** $(1\times1 + 1\times1 + 1\times1 + 1\times1) = 4$
4.  **Sum Channels:** $2 + 2 + 4 = 8$
5.  **Add Bias:** $8 + b_1 = 8 + 1 = \mathbf{9}$

**Full Feature Map 1 ($b_1=1$):**

```
[[9, 5],
 [5, 9]]
```

**Filter 2 ($W_2$) Weights & Bias ($b_2=0$):**

- $W_{2R}: [[0,1],[1,0]]$, $W_{2G}: [[1,0],[0,1]]$, $W_{2B}: [[0,0],[0,0]]$

**Full Feature Map 2 ($b_2=0$):**

```
[[0, 4],
 [4, 0]]
```

**Output Volume:**
Since we have 2 filters, the output will be $2 \times 2 \times 2$.

- **Feature Map 1:** Results from Filter 1.
- **Feature Map 2:** Results from Filter 2.

The final output tensor shape is $(H_{out}, W_{out}, Filters)$, which in this case is $2 \times 2 \times 2$.

The final operation for any single point in a feature map is:
$$Output = \left( \sum_{c=1}^{Channels} Input_c * Filter_c \right) + Bias$$
