# UNET Model Normalization and Time Embedding Notes

## Time Embedding Normalization

In the `sample_ddpm` function (used for generation/sampling), the time step `i` (which ranges from `timesteps` down to 1) is normalized before being passed to the model:

```python
t = torch.tensor([i / timesteps])[:, None, None, None].to(device)
```

This is consistent with the `ContextUnet.forward` method or how `EmbedFC` might expect inputs, ensuring the time signal is scaled between 0 and 1 (or close to it) rather than being raw integers up to 1000. This helps with the stability and scaling of the neural network inputs.

Interestingly, in the training loop:

```python
t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
# ...
pred_noise = nn_model(x_pert, t / timesteps)
```

The integer `t` is sampled randomly, but it is **also divided by `timesteps`** before being passed to `nn_model`. This confirms that the model always expects normalized time coordinates $t \in (0, 1]$.

## Image Normalization

### Data Loading (Training)

The training process normalizes input images to the range $[-1, 1]$. This is explicitly defined in `utils.py`:

```python
transform = transforms.Compose([
    transforms.ToTensor(),                # Converts [0,255] to range [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,))  # Normalizes to range [-1, 1] via (x - 0.5) / 0.5
])
```

This ensures the input images $x$ are centered at 0, matching the standard Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$ that is mixed in during the diffusion process (`perturb_input`). This symmetry is crucial for the stability of the diffusion training objective.

### Sampling / Plotting

While the model operates in the $[-1, 1]$ range, standard plotting libraries expect images in $[0, 1]$ or $[0, 255]$. The `utils.py` file handles this conversion in the `plot_sample` function using a utility called `unorm` (unity norm) and `norm_all`:

```python
def unorm(x):
    # unity norm. results in range of [0,1]
    # assume x (h,w,3)
    xmax = x.max((0,1))
    xmin = x.min((0,1))
    return(x - xmin)/(xmax - xmin)
```

This function dynamically scales the generated samples (which are approximately in $[-1, 1]$) back to $[0, 1]$ based on their actual min/max values, ensuring they display correctly without clipping or distortion.