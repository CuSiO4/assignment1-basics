import torch 
from einops import rearrange, einsum
import einx
channel_last = torch.randn(64,32,32,3)
B = torch.randn(32* 32, 32* 32)
height = width = 32
channel_first = rearrange(channel_last,"b h w c -> b c (h w)")
channel_first_transformed = einsum(channel_first, B,"b c pixel_in, pixel_in pixel_out -> b c pixel_out")
channel_out_transformed = rearrange(channel_first_transformed,"b c (h w)  -> b h w c",h = height,w = width)
print(channel_out_transformed.shape)
# channels_last_transformed = einx.dot(
#    " batch row_in col_in channel,(row_out col_out) (row_in col_in)"
#    "->batch row_out col_out channel",
#    channels_last, B,
#    col_in = width, col_out = width
# )
# print(channels_last_transformed.shape)
help(torch.nn.init.trunc_normal_)