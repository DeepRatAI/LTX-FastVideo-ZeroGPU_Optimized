# 🎬 DeepRat LTX Video - AI Video Generation

<div align="center">

![DeepRat Banner](https://img.shields.io/badge/DeepRat-LTX%20Video-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge)

**Advanced AI-powered video generation using LTX Video model**

[🚀 Try Demo](https://huggingface.co/spaces/DeepRat/LTX-Video-ZeroGPU-Optimized) | [📖 Documentation](#documentation) | [🎨 Examples](#examples)

</div>

---

<div align="center">
  <video src="https://github.com/user-attachments/assets/2b461957-50b5-4854-9a15-e6fc9ca05d85"
         controls playsinline muted loop width="768"></video>
</div>

---

## ✨ Features

* 🎬 **Text-to-Video**: Generate videos from text descriptions
* 🖼️ **Image-to-Video**: Animate static images with AI
* 🎞️ **Video-to-Video**: Transform and enhance existing videos
* 🎯 **Multi-Conditioning**: Apply multiple conditions at specific frames
* ⚡ **High Performance**: Optimized for CUDA with CPU fallback
* 🎨 **Flexible Control**: Fine-tune every aspect of generation

---

## 📸 Examples

### Text-to-Video (PtV)

| Entrada                                                                                                                                                               | Salida                                                                                                                                                                      |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <div> <strong>Example 1 — Majestic Black Lion</strong><br><em>Seed:</em> 1363812591<br><em>Prompt:</em> A beautiful and powerful black lion </div>                    | <div align="center"><video src="https://github.com/user-attachments/assets/8c84737d-9225-4784-8720-12154a6d553e" controls playsinline muted loop width="384"></video></div> |
| <div> <strong>Example 2 — Snow-Capped Mountains</strong><br><em>Seed:</em> 3804031196<br><em>Prompt:</em> A view from above of beautiful snow-capped mountains </div> | <div align="center"><video src="https://github.com/user-attachments/assets/c9011988-8c69-43fe-aab9-84f923fd2751" controls playsinline muted loop width="384"></video></div> |
| <div> <strong>Example 3 — Tiger Attacking Wild Boar</strong><br><em>Seed:</em> 1397763684<br><em>Prompt:</em> A tiger jumping/attacking a wild boar </div>            | <div align="center"><video src="https://github.com/user-attachments/assets/fa5dffdd-e2d5-4b1c-be6c-278180095672" controls playsinline muted loop width="384"></video></div> |

---

### Image-to-Video (ItV)

| Entrada (Imagen)                                                                                                                                                                                                                                                                                                                                                    | Salida (Video)                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <div align="center"><img alt="1830526882" src="https://github.com/user-attachments/assets/6bdeb80d-13c6-4db6-9859-51aac7a69599" width="384" /><br><sub><em>Seed:</em> 1830526882</sub><br><sub><em>Prompt:</em> the two cats in the image are fighting each other with kicks and Muay Thai fists in a very active and dizzying way like an action fight</sub></div> | <div align="center"><video src="https://github.com/user-attachments/assets/8b11b10c-f8a0-440e-a7a4-11f72081eda8" controls playsinline muted loop width="384"></video></div> |
| <div align="center"><img alt="738317591" src="https://github.com/user-attachments/assets/28a8a650-fa4c-4842-b460-384c69120341" width="384" /><br><sub><em>Seed:</em> 738317591</sub><br><sub><em>Prompt:</em> The cats from the picture are boxing each other agresivelly</sub></div>                                                                               | <div align="center"><video src="https://github.com/user-attachments/assets/81318049-2061-4201-9dcb-5736462447fa" controls playsinline muted loop width="384"></video></div> |
| <div align="center"><img alt="3858595085" src="https://github.com/user-attachments/assets/22391711-cdcd-495d-a2f3-d76c9bf11117" width="384" /><br><sub><em>Seed:</em> 3858595085</sub><br><sub><em>Prompt:</em> take the lion from the drawing and remove the background. turns the lion into a 3d model</sub></div>                                                | <div align="center"><video src="https://github.com/user-attachments/assets/241d6c40-2a0e-4428-a075-7d245461d226" controls playsinline muted loop width="384"></video></div> |
| <div align="center"><img alt="4273030543" src="https://github.com/user-attachments/assets/e26d5156-e1c6-4d0a-87e8-69fa710c9503" width="384" /><br><sub><em>Seed:</em> 4273030543</sub><br><sub><em>Prompt:</em> Make the skater in the image suffer a fall</sub></div>                                                                                              | <div align="center"><video src="https://github.com/user-attachments/assets/b94db5e9-66c0-478f-aaa5-d9f3d19df2b6" controls playsinline muted loop width="384"></video></div> |
| <div align="center"><img alt="1747446564" src="https://github.com/user-attachments/assets/53bad1d3-d8fe-4002-8f87-91b548ab05fb" width="384" /><br><sub><em>Seed:</em> 1747446564</sub><br><sub><em>Prompt:</em> The monkey takes a cool look and then puts on sunglasses</sub></div>                                                                                | <div align="center"><video src="https://github.com/user-attachments/assets/51533440-9b85-4c75-b30c-e25fdb40e658" controls playsinline muted loop width="384"></video></div> |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/DeepRatAI/LTX-FastVideo-ZeroGPU_Optimized.git
cd LTX-FastVideo-ZeroGPU_Optimized

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

| Parameter               | Description         | Default | Range       |
| ----------------------- | ------------------- | ------- | ----------- |
| `--height`              | Output video height | 704     | 256-720     |
| `--width`               | Output video width  | 1216    | 256-1280    |
| `--num_frames`          | Number of frames    | 121     | 1-257       |
| `--frame_rate`          | FPS of output       | 30      | 1-60        |
| `--seed`                | Random seed         | 171198  | Any integer |
| `--guidance_scale`      | Prompt adherence    | 3.0     | 1.0-20.0    |
| `--num_inference_steps` | Quality steps       | 50      | 1-100       |

### Conditioning Parameters

* `--conditioning_media_paths`: Path(s) to conditioning images/videos
* `--conditioning_strengths`: Strength of each condition (0.0-1.0)
* `--conditioning_start_frames`: Frame index where condition starts

---

## 📊 Technical Details

### Model Architecture

* **Base Model**: LTX Video (Lightricks)
* **Precision**: Mixed (BF16/FP32)
* **VAE**: Causal Video Autoencoder
* **Transformer**: 3D Transformer with symmetric patchifier
* **Scheduler**: Rectified Flow

### System Requirements

* **GPU**: NVIDIA GPU with 16GB+ VRAM (recommended)
* **RAM**: 32GB+ recommended
* **Storage**: 50GB+ for models
* **Python**: 3.10+

### Supported Resolutions

* **Width**: 256px - 1280px (divisible by 32)
* **Height**: 256px - 720px (divisible by 32)
* **Frames**: 1 - 257 (formula: N * 8 + 1)

---

## 💡 Tips for Best Results

### Text-to-Video

* Be specific about motion, lighting, and camera movement
* Use descriptive language: “slowly”, “dramatic”, “cinematic”
* Start with lower resolutions for faster iteration
* Avoid overly complex or contradictory prompts

### Image-to-Video

* Use conditioning strength 0.7–0.9 for natural motion
* Clear, high-quality input images work best
* Describe the desired motion explicitly
* Avoid very low conditioning strength (<0.5)

### General Tips

* Use negative prompts to avoid unwanted elements
* Adjust guidance scale: lower (2–4) for creativity, higher (5–8) for accuracy
* More inference steps = better quality but slower
* Use consistent seeds to reproduce results

---

## 📁 Project Structure

```
LTX-FastVideo-ZeroGPU_Optimized/
├── inference.py              # Main inference script
├── app.py                    # Gradio web interface
├── configs/                  # Configuration files
│   └── ltxv-13b-0.9.7-dev.yaml
├── ltx_video/               # Core LTX Video modules
│   ├── models/
│   ├── pipelines/
│   └── schedulers/
├── examples/                # Example outputs
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

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Credits & Acknowledgments

* **LTX Video**: [Lightricks](https://github.com/Lightricks/LTX-Video)
* **Model**: [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)
* **Paper**: [LTX-Video: Realtime Video Latent Diffusion](https://arxiv.org/abs/2411.17465)
* **Interface**: Built with Gradio
* **Maintainer**: DeepRat (solo)

---

## 🔗 Links

* 🌐 [Hugging Face Space](https://huggingface.co/spaces/DeepRat/LTX-Video-ZeroGPU-Optimized)
* 📦 [Model Card](https://huggingface.co/Lightricks/LTX-Video)
* 📖 [Original Repository](https://github.com/Lightricks/LTX-Video)
* 📄 [Research Paper](https://arxiv.org/abs/2411.17465)

---

## 🐛 Known Issues & Limitations

* Very high resolutions (>1280×720) require significant VRAM
* CPU inference is extremely slow (GPU strongly recommended)
* Long prompts (>77 tokens) may be truncated
* Some complex motions may not be fully captured

---

**Made with ❤️ by the DeepRat for the Community**

⭐ Star us on GitHub — it helps!

[⬆ Back to Top](#-deeprat-ltx-video---ai-video-generation)

</div>
