import torch
from einops import rearrange, einsum

images = torch.randn(64,128, 128, 3)
dim_by = torch.linspace(start = 0.0, end = 1.0, steps = 10)

dim_value = rearrange(dim_by,       "dim_value    -> 1 dim_value 1 1 1")
images_rearr = rearrange(images,    "b h w c      -> b 1 h w c")
dimmed_images = einsum(images, dim_by,"b h w c, dim_value -> b dim_value h w c")
print(dimmed_images.shape)