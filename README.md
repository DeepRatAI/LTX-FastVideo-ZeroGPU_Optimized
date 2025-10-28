# 🎬 DeepRat LTX Video - AI Video Generation

<div align="center">

![DeepRat Banner](https://img.shields.io/badge/DeepRat-LTX%20Video-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge)

**Advanced AI-powered video generation using LTX Video model**

[🚀 Try Demo](https://huggingface.co/spaces/YOUR_USERNAME/deeprat-ltx-video) | [📖 Documentation](#documentation) | [🎨 Examples](#examples)

</div>

---

## ✨ Features

- 🎬 **Text-to-Video**: Generate videos from text descriptions
- 🖼️ **Image-to-Video**: Animate static images with AI
- 🎞️ **Video-to-Video**: Transform and enhance existing videos
- 🎯 **Multi-Conditioning**: Apply multiple conditions at specific frames
- ⚡ **High Performance**: Optimized for CUDA with CPU fallback
- 🎨 **Flexible Control**: Fine-tune every aspect of generation

---

## 📸 Examples

### Picture-to-Video (PtV)

#### 1. Majestic Black Lion
<div align="center">
<img src="PtV/1363812591_a-beautiful-and-powerful-black-lion.jpg" width="600"/>
</div>

**Prompt**: `A beautiful and powerful black lion`  
**Seed**: `1363812591`  
**Type**: Text-to-Video

---

#### 2. Snow-Capped Mountains
<div align="center">
<img src="PtV/3804031196_a-view-from-above-of-beautiful-snow-capped-mountains.jpg" width="600"/>
</div>

**Prompt**: `A view from above of beautiful snow-capped mountains`  
**Seed**: `3804031196`  
**Type**: Text-to-Video

---

#### 3. Tiger vs Wild Boar
<div align="center">
<img src="PtV/1397763684_a-tiger-jumping-attacking-a-wild-boar.jpg" width="600"/>
</div>

**Prompt**: `A tiger jumping/attacking a wild boar`  
**Seed**: `1397763684`  
**Type**: Text-to-Video

---

### Image-to-Video (ItV)

#### 1. Muay Thai Cats Battle
<div align="center">
<img src="ItV/1830526882_the-two-cats-in-the-image-are-fighting-each-other-with-kicks-and-muay-thai-fists-in-a-very-active-and-dizzying-way-like-an-action-fight.jpg" width="600"/>
</div>

**Prompt**: `the two cats in the image are fighting each other with kicks and Muay Thai fists in a very active and dizzying way like an action fight`  
**Seed**: `1830526882`  
**Type**: Image-to-Video  
**Conditioning**: Full image conditioning

---

#### 2. Boxing Cats
<div align="center">
<img src="ItV/738317591_the-cats-from-the-picture-are-boxing-each-other-agresivelly.jpg" width="600"/>
</div>

**Prompt**: `The cats from the picture are boxing each other agresivelly`  
**Seed**: `738317591`  
**Type**: Image-to-Video  
**Conditioning**: Full image conditioning

---

#### 3. 3D Lion Transformation
<div align="center">
<img src="ItV/3858595085_take-the-lion-from-the-drawing-and-remove-the-background-turns-the-lion-into-a-3d-model.jpg" width="600"/>
</div>

**Prompt**: `take the lion from the drawing and remove the background. turns the lion into a 3d model`  
**Seed**: `3858595085`  
**Type**: Image-to-Video  
**Note**: 2D to 3D transformation

---

#### 4. Skater Fall
<div align="center">
<img src="ItV/4273030543_make-the-skater-in-the-image-suffer-a-fall.jpg" width="600"/>
</div>

**Prompt**: `Make the skater in the image suffer a fall`  
**Seed**: `4273030543`  
**Type**: Image-to-Video  
**Action**: Dynamic motion from static photo

---

#### 5. Cool Monkey with Sunglasses
<div align="center">
<img src="ItV/1747446564_the-monkey-takes-a-cool-look-and-then-puts-on-sunglasses.jpg" width="600"/>
</div>

**Prompt**: `The monkey takes a cool look and then puts on sunglasses`  
**Seed**: `1747446564`  
**Type**: Image-to-Video  
**Action**: Object interaction animation

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/deeprat-ltx-video.git
cd deeprat-ltx-video

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Text-to-Video
```bash
python inference.py \
  --prompt "A beautiful sunset over the ocean" \
  --height 704 \
  --width 1216 \
  --num_frames 121 \
  --seed 42
```

#### Image-to-Video
```bash
python inference.py \
  --prompt "The person in the image starts walking" \
  --conditioning_media_paths path/to/image.jpg \
  --conditioning_strengths 0.8 \
  --conditioning_start_frames 0 \
  --height 704 \
  --width 1216 \
  --num_frames 121 \
  --seed 42
```

---

## 🎛️ Parameters Guide

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--height` | Output video height | 704 | 256-720 |
| `--width` | Output video width | 1216 | 256-1280 |
| `--num_frames` | Number of frames | 121 | 1-257 |
| `--frame_rate` | FPS of output | 30 | 1-60 |
| `--seed` | Random seed | 171198 | Any integer |
| `--guidance_scale` | Prompt adherence | 3.0 | 1.0-20.0 |
| `--num_inference_steps` | Quality steps | 50 | 1-100 |

### Conditioning Parameters

- `--conditioning_media_paths`: Path(s) to conditioning images/videos
- `--conditioning_strengths`: Strength of each condition (0.0-1.0)
- `--conditioning_start_frames`: Frame index where condition starts

---

## 📊 Technical Details

### Model Architecture
- **Base Model**: LTX Video (Lightricks)
- **Precision**: Mixed (BF16/FP32)
- **VAE**: Causal Video Autoencoder
- **Transformer**: 3D Transformer with symmetric patchifier
- **Scheduler**: Rectified Flow

### System Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (recommended)
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ for models
- **Python**: 3.10+

### Supported Resolutions
- **Width**: 256px - 1280px (divisible by 32)
- **Height**: 256px - 720px (divisible by 32)
- **Frames**: 1 - 257 (formula: N * 8 + 1)

---

## 💡 Tips for Best Results

### Text-to-Video
- ✅ Be specific about motion, lighting, and camera movement
- ✅ Use descriptive language: "slowly", "dramatic", "cinematic"
- ✅ Start with lower resolutions for faster iteration
- ❌ Avoid overly complex or contradictory prompts

### Image-to-Video
- ✅ Use conditioning strength 0.7-0.9 for natural motion
- ✅ Clear, high-quality input images work best
- ✅ Describe the desired motion explicitly
- ❌ Don't use very low conditioning strength (<0.5)

### General Tips
- 🎯 Use negative prompts to avoid unwanted elements
- 🎯 Adjust guidance scale: lower (2-4) for creativity, higher (5-8) for accuracy
- 🎯 More inference steps = better quality but slower
- 🎯 Use consistent seeds to reproduce results

---

## 🔧 Advanced Configuration

### Config Files
The pipeline uses YAML configuration files located in `configs/`:

```yaml
# Example config
checkpoint_path: "ltxv-13b-0.9.7-dev.safetensors"
precision: "bfloat16"
sampler: "LinearQuadratic"
guidance_scale: 3.0
num_inference_steps: 50
```

### Custom Models
Place custom models in the `MODEL_DIR` and update config accordingly.

---

## 📁 Project Structure

```
deeprat-ltx-video/
├── inference.py              # Main inference script
├── app.py                    # Gradio web interface
├── configs/                  # Configuration files
│   └── ltxv-13b-0.9.7-dev.yaml
├── ltx_video/               # Core LTX Video modules
│   ├── models/
│   ├── pipelines/
│   └── schedulers/
├── Assets/                  # Example outputs
│   ├── PtV/                # Picture-to-Video examples
│   └── ItV/                # Image-to-Video examples
└── requirements.txt         # Python dependencies
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


---

## 🙏 Credits & Acknowledgments

- **LTX Video**: [Lightricks](https://github.com/Lightricks/LTX-Video)
- **Model**: [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)
- **Paper**: [LTX-Video: Realtime Video Latent Diffusion](https://arxiv.org/abs/2411.17465)
- **Interface**: Built with Gradio
- **Community**: DeepRat AI Community

---

## 🔗 Links

- 🌐 [Hugging Face Space](https://huggingface.co/spaces/YOUR_USERNAME/deeprat-ltx-video)
- 📦 [Model Card](https://huggingface.co/Lightricks/LTX-Video)
- 📖 [Original Repository](https://github.com/Lightricks/LTX-Video)


---

## 🐛 Known Issues & Limitations

- Very high resolutions (>1280x720) require significant VRAM
- CPU inference is extremely slow (GPU strongly recommended)
- Long prompts (>77 tokens) may be truncated
- Some complex motions may not be fully captured

---

## 📮 Contact

For questions, suggestions, or collaborations:

- GitHub Issues: [Create an issue](https://github.com/YOUR_USERNAME/deeprat-ltx-video/issues)
- Email: your.email@example.com
- Twitter: [@YourHandle](https://twitter.com/YourHandle)

---

<div align="center">

**Made with ❤️ by the DeepRat Community**

⭐ Star us on GitHub — it helps!

[⬆ Back to Top](#-deeprat-ltx-video---ai-video-generation)

</div>
[🚀 Try the Space](https://huggingface.co/spaces/DeepRatAI/ltx-video-zerogpu-optimized) • [📖 Documentation](docs/) • [🐛 Report Bug](https://github.com/DeepRatAI/LTX-FastVideo-ZeroGPU_Optimized/issues) • [✨ Request Feature](https://github.com/DeepRatAI/LTX-FastVideo-ZeroGPU_Optimized/issues)

</div>
