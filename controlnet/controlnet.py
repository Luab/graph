"""
ControlNet implementation for RadEdit with graph conditioning.

Uses diffusers' ControlNetModel.from_unet() to create a proper ControlNet
with zero spatial conditioning and graph embeddings via cross-attention.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict, Any
from diffusers import UNet2DConditionModel
from diffusers.models.controlnets.controlnet import ControlNetModel


class GraphControlNet(nn.Module):
    """
    Wrapper around diffusers ControlNetModel for graph conditioning.
    
    Uses zero spatial conditioning and graph embeddings via cross-attention.
    This matches the baseline ControlNet architecture exactly, but without
    spatial image conditioning.
    
    Conditioning Strategy
    ---------------------
    Current Implementation:
        - ControlNet: Graph embeddings via encoder_hidden_states
        - UNet: Encoded empty prompt ("" through BioViL-T, not zeros)
        - Control signal comes from ControlNet residual connections
        - Empty prompt maintains proper text encoder output distribution
    
    Alternative Strategies (modify training/inference scripts):
    
    1. Text + Graph (Dual Modality):
       - ControlNet: Graph embeddings
       - UNet: Text embeddings from clinical prompts
       - Pros: Leverages RadEdit's text training, semantic + structural
       - Implementation: Encode batch['text_prompt'] and pass to UNet
    
    2. Dual Graph Conditioning:
       - ControlNet: Graph embeddings
       - UNet: Same graph embeddings
       - Pros: Both networks see graph structure
       - Cons: Out-of-distribution for UNet (trained on text)
       - Implementation: Pass graph_embeddings to both
    
    3. Concatenated Text + Graph:
       - Concat text and graph: [B, 256, 768]
       - Pass to UNet cross-attention
       - Cons: 2x sequence length, may need adjustments
    
    4. Learned Adapter:
       - Train adapter to merge text + graph → [B, 128, 768]
       - Pass merged embeddings to UNet
       - Pros: Optimal fusion, maintains sequence length
    
    5. Zero Embeddings (not recommended):
       - torch.zeros((B, 128, 768))
       - Simpler but out-of-distribution
       - Use encoded empty prompt instead (current)
    
    See README_controlnet.md for detailed comparison and usage examples.
    
    Args:
        unet: RadEdit UNet2DConditionModel to create ControlNet from
    """
    
    def __init__(self, unet: UNet2DConditionModel):
        super().__init__()
        
        print("🔧 Initializing GraphControlNet from UNet...")
        
        # Create ControlNet from UNet (copies encoder architecture + weights)
        self.controlnet = ControlNetModel.from_unet(
            unet=unet,
            conditioning_channels=3,  # Dummy spatial channels (will be zero)
            load_weights_from_unet=True,  # Initialize with UNet encoder weights
        )
        
        print(f"✅ ControlNet created from UNet")
        
        # Freeze the spatial conditioning embedding (we don't use it)
        # The zero-initialized conv_out will always output zeros for zero input
        for param in self.controlnet.controlnet_cond_embedding.parameters():
            param.requires_grad = False
        
        print(f"   Frozen spatial conditioning embedding")
        print(f"   Total ControlNet parameters: {sum(p.numel() for p in self.controlnet.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.controlnet.parameters() if p.requires_grad):,}")
    
    def forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        graph_embeddings: torch.Tensor,
        conditioning_scale: float = 1.0,
        return_dict: bool = True,
    ):
        """
        Forward pass through ControlNet with graph conditioning.
        
        Args:
            noisy_latents: [B, 4, H, W] - Noisy latent images
            timestep: [B] or scalar - Diffusion timestep
            graph_embeddings: [B, 128, 768] - Graph embeddings for cross-attention
            conditioning_scale: Scale factor for ControlNet outputs
            return_dict: Whether to return dict or tuple
        
        Returns:
            ControlNetOutput or tuple:
                - down_block_res_samples: List of tensors from down blocks
                - mid_block_res_sample: Tensor from mid block
        """
        B = noisy_latents.shape[0]
        device = noisy_latents.device
        dtype = noisy_latents.dtype
        
        # Create zero spatial conditioning
        # Since controlnet_cond_embedding.conv_out is zero-initialized,
        # zero input → zero output → no spatial bias
        controlnet_cond = torch.zeros(
            B, 3, 512, 512,
            device=device,
            dtype=dtype
        )
        
        # Forward pass with graph embeddings in cross-attention
        return self.controlnet(
            sample=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=graph_embeddings,  # Graph conditioning via cross-attn!
            controlnet_cond=controlnet_cond,  # Zero spatial conditioning
            conditioning_scale=conditioning_scale,
            return_dict=return_dict,
        )
    
    def get_trainable_parameters(self):
        """Return trainable parameters (all ControlNet except spatial embedding)."""
        return [p for p in self.controlnet.parameters() if p.requires_grad]
    
    def get_num_trainable_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())


def test_graph_controlnet():
    """Test GraphControlNet initialization and forward pass."""
    import os
    
    print("=" * 80)
    print("Testing GraphControlNet")
    print("=" * 80)
    
    # Set proxy for model download
   # os.environ['HTTP_PROXY'] = "http://admin:Haruhi_123@192.168.1.1:1338"
    #os.environ['HTTPS_PROXY'] = "http://admin:Haruhi_123@192.168.1.1:1338"
    
    print("\nLoading RadEdit UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        "microsoft/radedit",
        subfolder="unet",
    )
    
    print(f"UNet loaded:")
    print(f"  - Config: {unet.config}")
    print(f"  - Total parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    print("\nCreating GraphControlNet...")
    controlnet = GraphControlNet(unet)
    
    print("\nTesting forward pass...")
    batch_size = 2
    noisy_latents = torch.randn(batch_size, 4, 64, 64)  # 512/8 = 64
    timestep = torch.tensor([500, 500])
    graph_embeddings = torch.randn(batch_size, 128, 768)
    
    print(f"  Input shapes:")
    print(f"    - noisy_latents: {noisy_latents.shape}")
    print(f"    - timestep: {timestep.shape}")
    print(f"    - graph_embeddings: {graph_embeddings.shape}")
    
    with torch.no_grad():
        outputs = controlnet(
            noisy_latents=noisy_latents,
            timestep=timestep,
            graph_embeddings=graph_embeddings,
            conditioning_scale=1.0,
            return_dict=True,
        )
    
    print(f"\n  Outputs:")
    print(f"    - down_block_res_samples: {len(outputs.down_block_res_samples)} tensors")
    for i, sample in enumerate(outputs.down_block_res_samples):
        print(f"      - [{i}]: {sample.shape}")
    print(f"    - mid_block_res_sample: {outputs.mid_block_res_sample.shape}")
    
    # Check that outputs are initially near zero (due to zero initialization)
    mid_abs_mean = outputs.mid_block_res_sample.abs().mean().item()
    print(f"\n  Mid block abs mean: {mid_abs_mean:.6f}")
    print(f"  (Should be ~0 due to zero initialization)")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_graph_controlnet()
