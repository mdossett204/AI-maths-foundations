# UnetDown Architecture

This block is the "Encoder" part of the network. Its job is to **process features** at the current resolution and then **shrink** the image to find higher-level abstract patterns.

## The Flow

```mermaid
graph TD
    Input["<b>Input Image/Feature</b><br/>(e.g., 32ch, 16x16)"]

    subgraph "UnetDown Block"
        direction TB

        %% Feature Processing
        Input --> ResBlock1["<b>ResidualConvBlock 1</b><br/>Extracts Features<br/>(Keeps size 16x16)"]
        ResBlock1 --> ResBlock2["<b>ResidualConvBlock 2</b><br/>Refines Features<br/>(Keeps size 16x16)"]

        %% The Output split
        ResBlock2 --> OutputFeatures["<b>Output for Skip Connection</b><br/>(Sent to UnetUp later)"]
        
        %% Downsampling
        ResBlock2 --> MaxPool["<b>MaxPool2d</b><br/>Kernel=2<br/><i>Halves Size: 16x16 -> 8x8</i>"]
    end

    MaxPool --> OutputNext["<b>Output to Next Layer</b><br/>(e.g., 64ch, 8x8)"]

    style Input fill:#e1f5fe,stroke:#01579b
    style OutputFeatures fill:#e8f5e9,stroke:#1b5e20
    style MaxPool fill:#ffccbc,stroke:#bf360c
    style ResBlock1 fill:#fff9c4,stroke:#fbc02d
    style ResBlock2 fill:#fff9c4,stroke:#fbc02d
```

## The Logic

1.  **Process (x2):** It runs two `ResidualConvBlock`s. This allows the network to "think" about the image at this specific resolution (16x16) and extract useful details (edges, textures).
2.  **Save for Later:** The output of these blocks is what gets sent across the "Skip Connection" to the `UnetUp` block later.
3.  **Shrink:** Finally, `nn.MaxPool2d(2)` looks at every 2x2 square of pixels and keeps only the biggest value. This cuts the width and height in half ($16 \rightarrow 8$), forcing the network to summarize the information.

## The Math: MaxPool2d (Downsampling)

The formula for calculating the output size is:
$$ H_{out} = \left\lfloor \frac{H_{in} + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} + 1 \right\rfloor $$

In `UnetDown`, the layer is `nn.MaxPool2d(2)`, which implies:
*   **Kernel Size ($k$):** 2
*   **Stride ($s$):** 2 (defaults to kernel size)
*   **Padding ($p$):** 0

**Example (16x16 input):**
$$ H_{out} = \left\lfloor \frac{16 + 0 - 2}{2} + 1 \right\rfloor = 8 $$

**What if there is a remainder?**
PyTorch's default behavior (`ceil_mode=False`) is to **drop** the last pixels if they don't fit into the kernel window. The $\lfloor \dots \rfloor$ (floor) operation handles this.

*   **Input 15:** $\lfloor \frac{15-2}{2} + 1 \rfloor = \lfloor 7.5 \rfloor = 7$ (The 15th pixel is ignored).

```