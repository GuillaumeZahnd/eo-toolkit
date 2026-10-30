import torch
import os
    
from mock_up_embedding_layer import compute_channel_stats, ImageStackEmbeddingMockUp


if __name__ == "__main__":

    # (T, C, D, W) T: time steps, C: channels, D: image depth, W: image width
    x = torch.rand(56, 16, 202, 202)

    # Per-channel statistics, NaN-aware
    mean, std = compute_channel_stats(x)

    net = ImageStackEmbeddingMockUp(
        in_channels=x.shape[1],
        embed_dim=128,
        patch_size=8,
        mean=mean,
        std=std
    )

    y = net(x)

    print(y.shape)  # (56, 128, 26, 26)
    
    os.makedirs("outputs", exist_ok=True)
    torch.save(y, "outputs/embeddings.pt")
