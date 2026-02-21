import torch
import torch.nn as nn
import torchvision.models as models

class VisionBackboneWrapper(nn.Module):
    """
    Wraps a ResNet18 model to output localized spatial feature maps 
    or a global pooled embedding depending on the C-JEPA projection strategy.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load ResNet18 weights
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_model = models.resnet18(weights=weights)
        
        # Strip the final fully connected layer and global average pooling layer.
        # This leaves us with the feature maps from layer4 
        # For 84x84 input, ResNet downsamples by 32x. 
        # 84 / 32 = 2.625 (ceil -> 3x3 spatial map).
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        self.feature_dim = 512
        
        # Add our own adaptive pool if we want a single flat feature vector per image
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor, return_spatial: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor
            return_spatial: if True, returns (B, 512, H', W') feature map
                            if False, returns (B, 512) flat vector
        """
        features = self.backbone(x)
        
        if return_spatial:
            return features
            
        pooled = self.pool(features)
        return pooled.reshape(pooled.size(0), -1)

# Quick test
if __name__ == "__main__":
    net = VisionBackboneWrapper(pretrained=False)
    dummy_input = torch.randn(2, 3, 84, 84)
    out_flat = net(dummy_input, return_spatial=False)
    out_spatial = net(dummy_input, return_spatial=True)
    
    print(f"Input: {dummy_input.shape}")
    print(f"Flat Output: {out_flat.shape}") # Should be (2, 512)
    print(f"Spatial Output: {out_spatial.shape}") # Should be (2, 512, 3, 3)
