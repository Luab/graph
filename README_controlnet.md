# ControlNet Graph Conditioning for RadEdit

This implementation adds graph-conditioned ControlNet to the RadEdit chest X-ray generation model, allowing control via medical report graph embeddings.

## Overview

- **Data**: CheXpert dataset with medical report graph embeddings ([128, 768] per sample)
- **Architecture**: Diffusers ControlNetModel adapted for graph conditioning via cross-attention
- **Training**: ControlNet encoder trainable, base UNet frozen
- **Conditioning**: Zero spatial conditioning + graph embeddings in cross-attention

## Quick Start

### 1. Data Preparation

Ensure you have:
- `reports_processed.csv` - CheXpert metadata (223,462 samples)
- `new_embeddings_expanded.h5` - Graph embeddings (223,462 x [128, 768])
- `/mnt/data/CheXpert/PNG/` - CheXpert images

### 2. Training

```bash
# Run training (default: 10 epochs, batch size 4, fp16)
./train_controlnet.sh

# Or with custom parameters:
python train_controlnet.py \
    --batch_size 8 \
    --num_epochs 20 \
    --learning_rate 1e-4 \
    --output_dir checkpoints/my_experiment
```

**Training Configuration**:
- Batch size: 4 (adjust based on GPU memory)
- Learning rate: 1e-4
- Mixed precision: FP16
- Gradient checkpointing: Enabled
- Gradient accumulation: 4 steps
- Optimizer: AdamW

**Expected Training Time** (on A100):
- ~10 hours per epoch (223K samples)
- Total: ~100 hours for 10 epochs

### 3. Inference

```bash
# Generate images from trained ControlNet
./inference_controlnet.sh checkpoints/controlnet/final/controlnet.pth

# Or with custom parameters:
python inference_controlnet.py \
    --controlnet_path checkpoints/controlnet/final/controlnet.pth \
    --graph_indices 0 1 2 3 4 5 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --conditioning_scale 1.0
```

## Architecture Details

### GraphControlNet

Uses `ControlNetModel.from_unet()` to create a trainable encoder copy:

```python
# Zero spatial conditioning (always outputs zeros)
controlnet_cond = torch.zeros(B, 3, 512, 512)

# Graph embeddings via cross-attention
down_block_res, mid_block_res = controlnet(
    sample=noisy_latents,
    timestep=timestep,
    encoder_hidden_states=graph_embeddings,  # [B, 128, 768]
    controlnet_cond=controlnet_cond,
)
```

**Trainable**: ControlNet encoder + zero convolutions  
**Frozen**: Spatial conditioning embedding, base UNet, VAE

---

## Conditioning Strategies - Complete Guide

### Current Implementation: Empty Prompt + Graph Control

**UNet (Frozen)**: Precomputed encoded empty prompt  
**ControlNet (Trainable)**: Graph embeddings

```
Empty string "" → BioViL-T → [1, 128, 768] (precomputed once, reused)
Graph Embedding → [B, 128, 768]
    ↓
ControlNet → residuals → UNet ← encoded ""
                           ↓
                      Generated Image
```

**Why this works:**
- ✅ Empty prompt keeps UNet in-distribution (proper BioViL-T output)
- ✅ Pure graph-based control via ControlNet residuals
- ✅ Efficient: Empty embeddings precomputed once
- ✅ Avoids out-of-distribution inputs to frozen UNet

**Code:**
```python
# Precompute once (in main()):
empty_prompt_embeds = text_encoder(tokenizer([""]), ...).last_hidden_state

# Use in training loop:
empty_embeddings = empty_prompt_embeds.repeat(batch_size, 1, 1)
noise_pred = unet(..., encoder_hidden_states=empty_embeddings, ...)
```

---

### Alternative Strategy 1: Text + Graph (Dual Modality)

**UNet**: Text embeddings from clinical prompts  
**ControlNet**: Graph embeddings

**Pros**: 
- Leverages RadEdit's text understanding
- Semantic (text) + Structural (graph) control
- More faithful to RadEdit's original training

**Cons**: 
- More complex
- Text encoding overhead during training

**Implementation:**
```python
# In training loop (modify train_controlnet.py line 275-282):
from transformers import AutoModel, AutoTokenizer

def encode_prompts(prompts, text_encoder, tokenizer, device):
    text_inputs = tokenizer(prompts, padding="max_length", 
                           max_length=128, truncation=True, 
                           return_tensors="pt")
    with torch.no_grad():
        embeds = text_encoder(
            input_ids=text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device),
        ).last_hidden_state
    return embeds

# In training loop:
text_embeddings = encode_prompts(
    batch['text_prompt'], text_encoder, tokenizer, device
)

noise_pred = unet(
    ...,
    encoder_hidden_states=text_embeddings,  # Text to UNet
    down_block_additional_residuals=down_block_res_samples,
    mid_block_additional_residual=mid_block_res_sample,
    ...
)
```

---

### Alternative Strategy 2: Dual Graph Conditioning

**Both UNet and ControlNet**: Graph embeddings

**Pros**: 
- Maximum graph conditioning
- Both networks learn from graph structure
- Simpler (no text encoding)

**Cons**: 
- Out-of-distribution for UNet (trained on text, not graphs)
- May cause unexpected behavior or mode collapse

**Implementation:**
```python
# In training loop (modify train_controlnet.py line 278):
noise_pred = unet(
    ...,
    encoder_hidden_states=graph_embeddings,  # Graph to UNet
    down_block_additional_residuals=down_block_res_samples,
    ...
)
```

---

### Alternative Strategy 3: Concatenated Text + Graph

Concatenate text and graph in sequence dimension.

**Pros**: 
- Simple fusion of both modalities
- Both signals available to UNet

**Cons**: 
- 2x sequence length: [B, 256, 768] vs [B, 128, 768]
- UNet trained on 128 tokens, may need adjustments
- Positional encoding mismatch

**Implementation:**
```python
# Get text embeddings [B, 128, 768]
text_embeds = encode_prompts(batch['text_prompt'], ...)

# Concatenate with graph [B, 128, 768]
combined = torch.cat([text_embeds, graph_embeddings], dim=1)  # [B, 256, 768]

# Pass to UNet
noise_pred = unet(
    ...,
    encoder_hidden_states=combined,  # [B, 256, 768]
    ...
)
```

**Note**: May require testing as sequence length differs from training.

---

### Alternative Strategy 4: Learned Adapter (Text+Graph Fusion)

Add trainable adapter to merge text and graph embeddings.

**Pros**: 
- Learns optimal fusion strategy
- Maintains [B, 128, 768] sequence length
- Flexible combination

**Cons**: 
- Additional trainable parameters
- More complex architecture

**Implementation:**
```python
class TextGraphAdapter(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, text_embeds, graph_embeds):
        # Cross-attention: text attends to graph
        attn_out, _ = self.cross_attn(text_embeds, graph_embeds, graph_embeds)
        text_embeds = self.norm1(text_embeds + attn_out)
        
        # FFN
        ffn_out = self.ffn(text_embeds)
        merged = self.norm2(text_embeds + ffn_out)
        
        return merged  # [B, 128, 768]

# In training:
adapter = TextGraphAdapter().to(device)  # Trainable
optimizer = AdamW(list(controlnet.parameters()) + list(adapter.parameters()), ...)

text_embeds = encode_prompts(batch['text_prompt'], ...)
merged_embeds = adapter(text_embeds, graph_embeddings)
noise_pred = unet(..., encoder_hidden_states=merged_embeds, ...)
```

---

### Alternative Strategy 5: Graph-to-Text Projection

Project graph embeddings into text embedding space.

**Pros**: 
- Maps graphs to text distribution UNet was trained on
- Single trainable projection layer

**Cons**: 
- May lose graph structural information
- Assumes text/graph spaces can be aligned

**Implementation:**
```python
class GraphToTextProjection(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        # Simple linear projection or transformer
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
    
    def forward(self, graph_embeds):
        return self.proj(graph_embeds)

# In training:
projection = GraphToTextProjection().to(device)  # Trainable
text_like_embeds = projection(graph_embeddings)
noise_pred = unet(..., encoder_hidden_states=text_like_embeds, ...)
```

---

### Alternative Strategy 6: Zero Embeddings (Not Recommended)

Use zero tensors instead of encoded empty prompt.

**Pros**: 
- Simplest implementation
- No text encoder needed

**Cons**: 
- ❌ Out-of-distribution for BioViL-T
- ❌ Missing positional encodings and structure
- ❌ May cause training instability
- ❌ Not proper text encoder output

**Implementation:**
```python
# Not recommended, but possible:
empty_embeddings = torch.zeros(
    (batch_size, 128, 768),
    device=device,
    dtype=dtype
)
noise_pred = unet(..., encoder_hidden_states=empty_embeddings, ...)
```

**Recommendation**: Use encoded empty prompt (current implementation) instead.

---

### Comparison Table

| Strategy | UNet Input | ControlNet Input | Complexity | Distribution Match | Trainable Extra | Recommended Use |
|----------|-----------|------------------|------------|-------------------|-----------------|-----------------|
| **Empty Prompt + Graph** (Current) | Encoded "" | Graph | Low | ✅ Yes | None | Default, pure graph control |
| Text + Graph | Encoded text | Graph | Medium | ✅ Yes | None | Dual modality, semantic control |
| Dual Graph | Graph | Graph | Low | ⚠️ No (UNet OOD) | None | Max graph conditioning |
| Concatenated | Text+Graph concat | Graph | Medium | ⚠️ 2x seq length | None | Simple multimodal |
| Adapter Fusion | Learned merge | Graph | High | ✅ Yes | Adapter | Optimal fusion |
| Graph→Text Proj | Projected graph | Graph | Medium | ✅ Maybe | Projection | Graph in text space |
| Zero Tensors | Zeros | Graph | Very Low | ❌ No | None | **Not recommended** |

---

### Switching Strategies

All strategies can be tested by modifying the training loop in `train_controlnet.py`:

1. **Current → Text+Graph**: Encode `batch['text_prompt']` and pass to UNet
2. **Current → Dual Graph**: Pass `graph_embeddings` to UNet (line ~278)
3. **Current → Zero Tensors**: Replace empty_embeddings with zeros (not recommended)
4. **Add Adapter/Projection**: Create new trainable module, modify optimizer

Example modification locations:
- Training: `train_controlnet.py` lines 270-285
- Validation: `train_controlnet.py` lines 360-380
- Inference: `inference_controlnet.py` lines 152-180

### Data Loading

`CheXpertGraphDataset` handles:
- 1:1 alignment between CSV rows and HDF5 embeddings
- Image preprocessing (512x512, normalize to [-1, 1])
- Text extraction from `section_impression`
- Train/valid split via path prefix

## File Structure

```
/home/luab/graph/
├── data_loader.py              # CheXpert dataset with graph embeddings
├── controlnet/
│   ├── __init__.py
│   └── controlnet.py           # GraphControlNet implementation
├── train_controlnet.py         # Training script
├── train_controlnet.sh         # Training launcher
├── inference_controlnet.py     # Inference script
├── inference_controlnet.sh     # Inference launcher
├── reports_processed.csv       # CheXpert metadata (223,462 samples)
├── new_embeddings_expanded.h5  # Graph embeddings [128, 768]
└── checkpoints/controlnet/     # Training checkpoints
    ├── checkpoint-epoch-1/
    ├── checkpoint-epoch-2/
    └── final/
        └── controlnet.pth
```

## Key Implementation Decisions

### 1. No Graph Encoder Needed
- Graph embeddings already [128, 768] (matches BioViL-T text encoder)
- Directly compatible with UNet cross-attention
- No additional projection required

### 2. Zero Spatial Conditioning
- `controlnet_cond_embedding` frozen
- Always outputs zeros (zero-initialized conv)
- No spatial bias, pure graph conditioning

### 3. Using Diffusers ControlNet
- Exact baseline architecture
- Proper time embeddings, residuals, zero convolutions
- Battle-tested implementation

### 4. Encoded Empty Prompt (Not Zeros)
- UNet receives encoded "" string through BioViL-T
- Precomputed once, reused throughout training
- Maintains proper text encoder output distribution
- More principled than zero tensors

### 5. Training Strategy
- Freeze base UNet (preserve RadEdit knowledge)
- Train only ControlNet encoder
- Graph embeddings to ControlNet cross-attention
- Empty prompt to UNet (unconditional)
- Standard DDPM loss on noise prediction

## Monitoring

Training logs to Weights & Biases:
- Project: `controlnet-graph-conditioning`
- Metrics: `train/loss`, `train/avg_loss`, `val/loss`
- Step frequency: Every 10 steps

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (default: 4 → 2)
- Enable `--gradient_checkpointing` (default: enabled)
- Increase `--gradient_accumulation_steps` (default: 4 → 8)

### Proxy Issues
Set in scripts:
```bash
export HTTP_PROXY="http://admin:Haruhi_123@192.168.1.1:1338"
export HTTPS_PROXY="http://admin:Haruhi_123@192.168.1.1:1338"
```

### Data Loading Errors
- Check image paths match PNG extension (CSV has .jpg, files are .png)
- Verify HDF5 index alignment with CSV rows
- Ensure train/valid split filtering works

## Citation

```bibtex
@article{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models},
  author={Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  journal={arXiv preprint arXiv:2302.05543},
  year={2023}
}

@article{microsoft2023radedit,
  title={RAdEdit: stress-testing biomedical vision models via diffusion image editing},
  author={Microsoft Research},
  year={2023}
}
```

## License

See main project LICENSE.

