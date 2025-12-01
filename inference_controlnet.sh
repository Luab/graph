#!/bin/bash
#
# Inference script for ControlNet with graph conditioning
#
# Usage: ./inference_controlnet.sh <controlnet_checkpoint_path>
#
# Example:
#   ./inference_controlnet.sh checkpoints/controlnet/final/controlnet.pth
#

set -e  # Exit on error

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Error: Please provide path to ControlNet checkpoint"
    echo "Usage: $0 <controlnet_checkpoint_path>"
    echo "Example: $0 checkpoints/controlnet/final/controlnet.pth"
    exit 1
fi

CONTROLNET_PATH="$1"

# Check if checkpoint exists
if [ ! -f "$CONTROLNET_PATH" ]; then
    echo "Error: ControlNet checkpoint not found at $CONTROLNET_PATH"
    exit 1
fi

# Set HTTP proxy for downloading models
export HTTP_PROXY="http://admin:Haruhi_123@192.168.1.1:1338"
export HTTPS_PROXY="http://admin:Haruhi_123@192.168.1.1:1338"

# Navigate to project directory
cd /home/luab/graph

# Activate virtual environment
source venv/bin/activate

# Check if embeddings exist
if [ ! -f "new_embeddings_expanded.h5" ]; then
    echo "Error: new_embeddings_expanded.h5 not found!"
    exit 1
fi

echo "================================"
echo "ControlNet Inference Configuration"
echo "================================"
echo "ControlNet checkpoint: $CONTROLNET_PATH"
echo "Graph embeddings: new_embeddings_expanded.h5"
echo "Output dir: outputs/controlnet"
echo "Num inference steps: 50"
echo "Guidance scale: 7.5"
echo "================================"
echo ""

# Run inference
python inference_controlnet.py \
    --controlnet_path "$CONTROLNET_PATH" \
    --graph_embeddings new_embeddings_expanded.h5 \
    --graph_indices 0 1 2 3 4 5 6 7 8 9 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --conditioning_scale 1.0 \
    --height 512 \
    --width 512 \
    --output_dir outputs/controlnet \
    --seed 42

echo ""
echo "✅ Inference complete! Check outputs/controlnet/ for generated images"

