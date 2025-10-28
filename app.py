import gradio as gr
import torch
import spaces
import numpy as np
import random
import os
import yaml
from pathlib import Path
import imageio
import tempfile
from PIL import Image
from huggingface_hub import hf_hub_download
import shutil

from inference import (
    create_ltx_video_pipeline,
    create_latent_upsampler,
    load_image_to_tensor_with_resize_and_crop,
    seed_everething,
    get_device,
    calculate_padding,
    load_media_file,
)
from ltx_video.pipelines.pipeline_ltx_video import (
    ConditioningItem,
    LTXMultiScalePipeline,
    LTXVideoPipeline,
)
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

config_file_path = "configs/ltxv-13b-0.9.8-distilled.yaml"
with open(config_file_path, "r") as file:
    PIPELINE_CONFIG_YAML = yaml.safe_load(file)

LTX_REPO = "Lightricks/LTX-Video"
MAX_IMAGE_SIZE = PIPELINE_CONFIG_YAML.get("max_resolution", 1280)
MAX_NUM_FRAMES = 257

FPS = 30.0

# Modelos globales
pipeline_instance = None
latent_upsampler_instance = None
models_dir = "downloaded_models_gradio_cpu_init"
Path(models_dir).mkdir(parents=True, exist_ok=True)

print("Downloading models (if not present)...")
distilled_model_actual_path = hf_hub_download(
    repo_id=LTX_REPO,
    filename=PIPELINE_CONFIG_YAML["checkpoint_path"],
    local_dir=models_dir,
    local_dir_use_symlinks=False,
)
PIPELINE_CONFIG_YAML["checkpoint_path"] = distilled_model_actual_path
print(f"Distilled model path: {distilled_model_actual_path}")

SPATIAL_UPSCALER_FILENAME = PIPELINE_CONFIG_YAML["spatial_upscaler_model_path"]
spatial_upscaler_actual_path = hf_hub_download(
    repo_id=LTX_REPO,
    filename=SPATIAL_UPSCALER_FILENAME,
    local_dir=models_dir,
    local_dir_use_symlinks=False,
)
PIPELINE_CONFIG_YAML["spatial_upscaler_model_path"] = spatial_upscaler_actual_path
print(f"Spatial upscaler model path: {spatial_upscaler_actual_path}")

print("Creating LTX Video pipeline on CPU...")
pipeline_instance = create_ltx_video_pipeline(
    ckpt_path=PIPELINE_CONFIG_YAML["checkpoint_path"],
    precision=PIPELINE_CONFIG_YAML["precision"],
    text_encoder_model_name_or_path=PIPELINE_CONFIG_YAML[
        "text_encoder_model_name_or_path"
    ],
    sampler=PIPELINE_CONFIG_YAML["sampler"],
    device="cpu",
    enhance_prompt=False,
    prompt_enhancer_image_caption_model_name_or_path=PIPELINE_CONFIG_YAML[
        "prompt_enhancer_image_caption_model_name_or_path"
    ],
    prompt_enhancer_llm_model_name_or_path=PIPELINE_CONFIG_YAML[
        "prompt_enhancer_llm_model_name_or_path"
    ],
)
print("LTX Video pipeline created on CPU.")

if PIPELINE_CONFIG_YAML.get("spatial_upscaler_model_path"):
    print("Creating latent upsampler on CPU...")
    latent_upsampler_instance = create_latent_upsampler(
        PIPELINE_CONFIG_YAML["spatial_upscaler_model_path"], device="cpu"
    )
    print("Latent upsampler created on CPU.")

target_inference_device = "cuda"
print(f"Target inference device: {target_inference_device}")
pipeline_instance.to(target_inference_device)
if latent_upsampler_instance:
    latent_upsampler_instance.to(target_inference_device)


# Cálculo de dimensiones
MIN_DIM_SLIDER = 256
TARGET_FIXED_SIDE = 768


def calculate_new_dimensions(orig_w, orig_h):
    if orig_w == 0 or orig_h == 0:
        return int(TARGET_FIXED_SIDE), int(TARGET_FIXED_SIDE)

    if orig_w >= orig_h:
        new_h = TARGET_FIXED_SIDE
        aspect_ratio = orig_w / orig_h
        new_w_ideal = new_h * aspect_ratio

        new_w = round(new_w_ideal / 32) * 32

        new_w = max(MIN_DIM_SLIDER, min(new_w, MAX_IMAGE_SIZE))
        new_h = max(MIN_DIM_SLIDER, min(new_h, MAX_IMAGE_SIZE))
    else:
        new_w = TARGET_FIXED_SIDE
        aspect_ratio = orig_h / orig_w
        new_h_ideal = new_w * aspect_ratio

        new_h = round(new_h_ideal / 32) * 32

        new_h = max(MIN_DIM_SLIDER, min(new_h, MAX_IMAGE_SIZE))
        new_w = max(MIN_DIM_SLIDER, min(new_w, MAX_IMAGE_SIZE))

    return int(new_h), int(new_w)


def get_duration(
    prompt,
    negative_prompt,
    input_image_filepath,
    input_video_filepath,
    height_ui,
    width_ui,
    mode,
    duration_ui,
    ui_frames_to_use,
    seed_ui,
    randomize_seed,
    ui_guidance_scale,
    improve_texture_flag,
    slow_motion_flag,
    progress,
):
    # Cálculo dinámico optimizado para ZeroGPU
    # Frames estimados basado en duración
    estimated_frames = int(duration_ui * FPS)

    # Factor de resolución (pixeles totales / millón)
    resolution_factor = (height_ui * width_ui) / 1_000_000

    # Base: ~0.4s por frame en resolución estándar (512x704)
    base_time_per_frame = 0.4

    # Ajuste por resolución
    time_per_frame = base_time_per_frame * (1 + resolution_factor * 0.3)

    # Tiempo estimado para single pass
    single_pass_time = estimated_frames * time_per_frame

    # Si multi-scale está activado, duplicar tiempo + overhead
    if improve_texture_flag:
        total_time = single_pass_time * 2.2
    else:
        total_time = single_pass_time

    # Añadir overhead de carga/descarga de modelo (10s)
    total_time += 10

    # Redondear hacia arriba y añadir margen de seguridad (15%)
    final_duration = int(total_time * 1.15)

    # Límite mínimo y máximo
    final_duration = max(30, min(final_duration, 120))

    return final_duration


@spaces.GPU(duration=get_duration)
def generate(
    prompt,
    negative_prompt,
    input_image_filepath=None,
    input_video_filepath=None,
    height_ui=512,
    width_ui=704,
    mode="text-to-video",
    duration_ui=2.0,
    ui_frames_to_use=9,
    seed_ui=42,
    randomize_seed=True,
    ui_guidance_scale=3.0,
    improve_texture_flag=True,
    slow_motion_flag=False,
    progress=gr.Progress(track_tqdm=True),
):
    if mode == "image-to-video":
        if not input_image_filepath:
            raise gr.Error("input_image_filepath is required for image-to-video mode")
    elif mode == "video-to-video":
        if not input_video_filepath:
            raise gr.Error("input_video_filepath is required for video-to-video mode")
    elif mode == "text-to-video":
        pass
    else:
        raise gr.Error(
            f"Invalid mode: {mode}. Must be one of: text-to-video, image-to-video, video-to-video"
        )

    if randomize_seed:
        seed_ui = random.randint(0, 2**32 - 1)
    seed_everething(int(seed_ui))

    # Procesar slow motion
    processed_prompt = prompt
    processed_negative_prompt = negative_prompt
    
    if slow_motion_flag:
        # Activar slow motion explícitamente
        if "slow motion" not in prompt.lower() and "slow-mo" not in prompt.lower():
            processed_prompt = f"{prompt}, slow motion, cinematic slow-mo, slowed down"
    else:
        # Desactivar slow motion: forzar velocidad normal
        # Añadir términos de velocidad normal al prompt positivo (más efectivo)
        if "real-time" not in prompt.lower() and "normal speed" not in prompt.lower():
            processed_prompt = f"{prompt}, real-time speed, normal motion speed, fast-paced"
        # Y también al negative prompt por seguridad
        if "slow motion" not in negative_prompt.lower() and "slow-mo" not in negative_prompt.lower():
            processed_negative_prompt = f"{negative_prompt}, slow motion, slow-mo, very slow, slowed down, ultra slow motion"

    target_frames_ideal = duration_ui * FPS
    target_frames_rounded = round(target_frames_ideal)
    if target_frames_rounded < 1:
        target_frames_rounded = 1

    n_val = round((float(target_frames_rounded) - 1.0) / 8.0)
    actual_num_frames = int(n_val * 8 + 1)

    actual_num_frames = max(9, actual_num_frames)
    actual_num_frames = min(MAX_NUM_FRAMES, actual_num_frames)

    actual_height = int(height_ui)
    actual_width = int(width_ui)

    height_padded = ((actual_height - 1) // 32 + 1) * 32
    width_padded = ((actual_width - 1) // 32 + 1) * 32
    num_frames_padded = ((actual_num_frames - 2) // 8 + 1) * 8 + 1
    if num_frames_padded != actual_num_frames:
        print(
            f"Warning: actual_num_frames ({actual_num_frames}) and num_frames_padded ({num_frames_padded}) differ. Using num_frames_padded for pipeline."
        )

    padding_values = calculate_padding(
        actual_height, actual_width, height_padded, width_padded
    )

    call_kwargs = {
        "prompt": processed_prompt,
        "negative_prompt": processed_negative_prompt,
        "height": height_padded,
        "width": width_padded,
        "num_frames": num_frames_padded,
        "frame_rate": int(FPS),
        "generator": torch.Generator(device=target_inference_device).manual_seed(
            int(seed_ui)
        ),
        "output_type": "pt",
        "conditioning_items": None,
        "media_items": None,
        "decode_timestep": PIPELINE_CONFIG_YAML["decode_timestep"],
        "decode_noise_scale": PIPELINE_CONFIG_YAML["decode_noise_scale"],
        "stochastic_sampling": PIPELINE_CONFIG_YAML["stochastic_sampling"],
        "image_cond_noise_scale": 0.15,
        "is_video": True,
        "vae_per_channel_normalize": True,
        "mixed_precision": (PIPELINE_CONFIG_YAML["precision"] == "mixed_precision"),
        "offload_to_cpu": False,
        "enhance_prompt": False,
    }

    stg_mode_str = PIPELINE_CONFIG_YAML.get("stg_mode", "attention_values")
    if stg_mode_str.lower() in ["stg_av", "attention_values"]:
        call_kwargs["skip_layer_strategy"] = SkipLayerStrategy.AttentionValues
    elif stg_mode_str.lower() in ["stg_as", "attention_skip"]:
        call_kwargs["skip_layer_strategy"] = SkipLayerStrategy.AttentionSkip
    elif stg_mode_str.lower() in ["stg_r", "residual"]:
        call_kwargs["skip_layer_strategy"] = SkipLayerStrategy.Residual
    elif stg_mode_str.lower() in ["stg_t", "transformer_block"]:
        call_kwargs["skip_layer_strategy"] = SkipLayerStrategy.TransformerBlock
    else:
        raise ValueError(f"Invalid stg_mode: {stg_mode_str}")

    if mode == "image-to-video" and input_image_filepath:
        try:
            media_tensor = load_image_to_tensor_with_resize_and_crop(
                input_image_filepath, actual_height, actual_width
            )
            media_tensor = torch.nn.functional.pad(media_tensor, padding_values)
            call_kwargs["conditioning_items"] = [
                ConditioningItem(media_tensor.to(target_inference_device), 0, 1.0)
            ]
        except Exception as e:
            print(f"Error loading image {input_image_filepath}: {e}")
            raise gr.Error(f"Could not load image: {e}")
    elif mode == "video-to-video" and input_video_filepath:
        try:
            call_kwargs["media_items"] = load_media_file(
                media_path=input_video_filepath,
                height=actual_height,
                width=actual_width,
                max_frames=int(ui_frames_to_use),
                padding=padding_values,
            ).to(target_inference_device)
        except Exception as e:
            print(f"Error loading video {input_video_filepath}: {e}")
            raise gr.Error(f"Could not load video: {e}")

    print(
        f"Moving models to {target_inference_device} for inference (if not already there)..."
    )

    active_latent_upsampler = None
    if improve_texture_flag and latent_upsampler_instance:
        active_latent_upsampler = latent_upsampler_instance

    result_images_tensor = None
    if improve_texture_flag:
        if not active_latent_upsampler:
            raise gr.Error(
                "Spatial upscaler model not loaded or improve_texture not selected, cannot use multi-scale."
            )

        multi_scale_pipeline_obj = LTXMultiScalePipeline(
            pipeline_instance, active_latent_upsampler
        )

        first_pass_args = PIPELINE_CONFIG_YAML.get("first_pass", {}).copy()
        first_pass_args["guidance_scale"] = float(ui_guidance_scale)
        first_pass_args.pop("num_inference_steps", None)

        second_pass_args = PIPELINE_CONFIG_YAML.get("second_pass", {}).copy()
        second_pass_args["guidance_scale"] = float(ui_guidance_scale)
        first_pass_args.pop("num_inference_steps", None)

        multi_scale_call_kwargs = call_kwargs.copy()
        multi_scale_call_kwargs.update(
            {
                "downscale_factor": PIPELINE_CONFIG_YAML["downscale_factor"],
                "first_pass": first_pass_args,
                "second_pass": second_pass_args,
            }
        )

        print(
            f"Calling multi-scale pipeline (eff. HxW: {actual_height}x{actual_width}, Frames: {actual_num_frames} -> Padded: {num_frames_padded}) on {target_inference_device}"
        )
        result_images_tensor = multi_scale_pipeline_obj(
            **multi_scale_call_kwargs
        ).images
    else:
        single_pass_call_kwargs = call_kwargs.copy()
        first_pass_config_from_yaml = PIPELINE_CONFIG_YAML.get("first_pass", {})

        single_pass_call_kwargs["timesteps"] = first_pass_config_from_yaml.get(
            "timesteps"
        )
        single_pass_call_kwargs["guidance_scale"] = float(ui_guidance_scale)
        single_pass_call_kwargs["stg_scale"] = first_pass_config_from_yaml.get(
            "stg_scale"
        )
        single_pass_call_kwargs["rescaling_scale"] = first_pass_config_from_yaml.get(
            "rescaling_scale"
        )
        single_pass_call_kwargs["skip_block_list"] = first_pass_config_from_yaml.get(
            "skip_block_list"
        )

        single_pass_call_kwargs.pop("num_inference_steps", None)
        single_pass_call_kwargs.pop("first_pass", None)
        single_pass_call_kwargs.pop("second_pass", None)
        single_pass_call_kwargs.pop("downscale_factor", None)

        print(
            f"Calling base pipeline (padded HxW: {height_padded}x{width_padded}, Frames: {actual_num_frames} -> Padded: {num_frames_padded}) on {target_inference_device}"
        )
        result_images_tensor = pipeline_instance(**single_pass_call_kwargs).images

    if result_images_tensor is None:
        raise gr.Error("Generation failed.")

    pad_left, pad_right, pad_top, pad_bottom = padding_values
    slice_h_end = -pad_bottom if pad_bottom > 0 else None
    slice_w_end = -pad_right if pad_right > 0 else None

    result_images_tensor = result_images_tensor[
        :, :, :actual_num_frames, pad_top:slice_h_end, pad_left:slice_w_end
    ]

    video_np = result_images_tensor[0].permute(1, 2, 3, 0).cpu().float().numpy()

    video_np = np.clip(video_np, 0, 1)
    video_np = (video_np * 255).astype(np.uint8)

    temp_dir = tempfile.mkdtemp()
    timestamp = random.randint(10000, 99999)
    output_video_path = os.path.join(temp_dir, f"output_{timestamp}.mp4")

    try:
        with imageio.get_writer(
            output_video_path, fps=call_kwargs["frame_rate"], macro_block_size=1
        ) as video_writer:
            for frame_idx in range(video_np.shape[0]):
                progress(frame_idx / video_np.shape[0], desc="Saving video")
                video_writer.append_data(video_np[frame_idx])
    except Exception as e:
        print(f"Error saving video with macro_block_size=1: {e}")
        try:
            with imageio.get_writer(
                output_video_path,
                fps=call_kwargs["frame_rate"],
                format="FFMPEG",
                codec="libx264",
                quality=8,
            ) as video_writer:
                for frame_idx in range(video_np.shape[0]):
                    progress(
                        frame_idx / video_np.shape[0],
                        desc="Saving video (fallback ffmpeg)",
                    )
                    video_writer.append_data(video_np[frame_idx])
        except Exception as e2:
            print(f"Fallback video saving error: {e2}")
            raise gr.Error(f"Failed to save video: {e2}")

    return output_video_path, seed_ui


def update_task_image():
    return "image-to-video"


def update_task_text():
    return "text-to-video"


def update_task_video():
        return "video-to-video"


css = """
/* === Color Palette === */
:root {
    --primary-bg: #1a0d2e;
    --secondary-bg: #2a1548;
    --accent-purple: #8b5cf6;
    --accent-pink: #ec4899;
    --neon-purple: #a855f7;
    --text-light: #e9d5ff;
    --border-color: #6b21a8;
}

/* === Global Container === */
#col-container {
    margin: 0 auto;
    max-width: 1400px;
}

/* === Main Background === */
body, .gradio-container {
    background: linear-gradient(135deg, #1a0d2e 0%, #2a1548 100%) !important;
}

/* === Custom Header === */
.custom-header {
    text-align: center;
    padding: 30px 20px;
    background: linear-gradient(135deg, #2a1548 0%, #1a0d2e 100%);
    border-radius: 16px;
    border: 2px solid var(--border-color);
    margin-bottom: 25px;
    box-shadow: 0 8px 32px rgba(139, 92, 246, 0.3);
    position: relative;
    z-index: 100;
}

.custom-header h1 {
    font-family: 'Courier New', monospace;
    font-size: 3rem;
    font-weight: bold;
    color: var(--neon-purple);
    letter-spacing: 8px;
    margin-bottom: 10px;
    position: relative;
    z-index: 10;
    text-shadow: 0 0 20px rgba(168, 85, 247, 0.8), 
                 0 0 40px rgba(236, 72, 153, 0.5);
}

.custom-header p {
    color: var(--text-light);
    font-size: 1.1rem;
    margin: 8px 0;
    opacity: 0.9;
    position: relative;
    z-index: 10;
}

.custom-header .highlight {
    background: rgba(139, 92, 246, 0.2);
    padding: 12px 20px;
    border-radius: 8px;
    border: 1px solid var(--border-color);
    margin-top: 15px;
    font-weight: 500;
    position: relative;
    z-index: 10;
}

/* === Tabs Styling === */
.tab-nav {
    background: var(--secondary-bg) !important;
    border-radius: 12px !important;
    border: 2px solid var(--border-color) !important;
    padding: 8px !important;
}

button.svelte-1b6s6s {
    background: transparent !important;
    color: var(--text-light) !important;
    border: none !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

button.svelte-1b6s6s.selected {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink)) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
}

/* === Input Fields === */
textarea, input[type="text"], input[type="number"] {
    background: rgba(42, 21, 72, 0.6) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-light) !important;
    padding: 12px !important;
    font-size: 1rem !important;
}

textarea:focus, input:focus {
    border-color: var(--neon-purple) !important;
    box-shadow: 0 0 15px rgba(168, 85, 247, 0.3) !important;
}

/* === Buttons === */
button.primary {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4) !important;
    transition: all 0.3s ease !important;
}

button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6) !important;
}

/* === Accordions === */
.accordion {
    background: rgba(42, 21, 72, 0.4) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 12px !important;
    margin: 15px 0 !important;
}

.accordion summary {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(236, 72, 153, 0.2)) !important;
    color: var(--text-light) !important;
    padding: 16px 20px !important;
    border-radius: 10px !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}

.accordion summary:hover {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.3), rgba(236, 72, 153, 0.3)) !important;
}

/* === Upload Areas === */
.upload-container {
    background: rgba(42, 21, 72, 0.3) !important;
    border: 2px dashed var(--border-color) !important;
    border-radius: 12px !important;
    padding: 30px !important;
    text-align: center !important;
    transition: all 0.3s ease !important;
}

.upload-container:hover {
    border-color: var(--neon-purple) !important;
    background: rgba(139, 92, 246, 0.1) !important;
}

/* === Output Video === */
video {
    border-radius: 12px !important;
    border: 2px solid var(--border-color) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
}

/* === Sliders === */
input[type="range"] {
    background: var(--secondary-bg) !important;
}

input[type="range"]::-webkit-slider-thumb {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink)) !important;
    box-shadow: 0 0 10px rgba(139, 92, 246, 0.5) !important;
}

/* === Checkboxes === */
input[type="checkbox"] {
    accent-color: var(--accent-purple) !important;
}

/* === Labels === */
label {
    color: var(--text-light) !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
}

/* === Tips/Info Boxes === */
.info-tip {
    background: rgba(139, 92, 246, 0.15);
    border-left: 4px solid var(--accent-purple);
    padding: 15px 20px;
    border-radius: 8px;
    margin: 20px 0;
    color: var(--text-light);
    font-size: 0.95rem;
    line-height: 1.6;
}

/* === Footer Links === */
a {
    color: var(--neon-purple) !important;
    text-decoration: none !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

a:hover {
    color: var(--accent-pink) !important;
    text-shadow: 0 0 10px rgba(236, 72, 153, 0.5) !important;
}

/* === Markdown Styling === */
.markdown-text {
    color: var(--text-light) !important;
}

/* === Responsive === */
@media (max-width: 768px) {
    .custom-header h1 {
        font-size: 2rem;
        letter-spacing: 4px;
    }
}
"""

with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:
    # Custom Header
    gr.HTML("""
    <div class="custom-header">
        <h1>LTX - VIDEO GENERATOR</h1>
        <p><em>Powered by LTX Video 0.9.8 13B Distilled</em></p>
        <p class="highlight">Now with "ZeroGPU Smart Configuration" allowing the generation of longer videos without timeout</p>
    </div>
    """)
    
    # Info Tips at top
    gr.Markdown("""
    <div class="info-tip">
        💡 <strong>Tip:</strong> The complexity of the prompt is not always reflected in better results; in general, it is advisable to keep the prompts descriptive but clear, consistent and simple
    </div>
    """)
    
    gr.Markdown("""
    <div class="info-tip">
        ⚡ <strong>Recommendation:</strong> I recommend activating ZeroGPU Smart Settings by default to avoid errors during timeout generation and conflicts with the use of ZeroGPU
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tab("Image to Video") as image_tab:
                video_i_hidden = gr.Textbox(label="video_i", visible=False, value=None)
                image_i2v = gr.Image(
                    label="Upload your image",
                    type="filepath",
                    sources=["upload", "webcam", "clipboard"],
                )
                i2v_prompt = gr.Textbox(
                    label="Prompt",
                    value="The creature from the image starts to move",
                    lines=3,
                )
                i2v_button = gr.Button("Generate Image-to-Video", variant="primary")
            with gr.Tab("Prompt to Video") as text_tab:
                image_n_hidden = gr.Textbox(label="image_n", visible=False, value=None)
                video_n_hidden = gr.Textbox(label="video_n", visible=False, value=None)
                t2v_prompt = gr.Textbox(
                    label="Prompt",
                    value="A majestic dragon flying over a medieval castle",
                    lines=3,
                )
                t2v_button = gr.Button("Generate Text-to-Video", variant="primary")
            with gr.Tab("video-to-video", visible=False) as video_tab:
                image_v_hidden = gr.Textbox(label="image_v", visible=False, value=None)
                video_v2v = gr.Video(
                    label="Input Video", sources=["upload", "webcam"]
                )  # type defaults to filepath
                frames_to_use = gr.Slider(
                    label="Frames to use from input video",
                    minimum=9,
                    maximum=MAX_NUM_FRAMES,
                    value=9,
                    step=8,
                    info="Number of initial frames to use for conditioning/transformation. Must be N*8+1.",
                )
                v2v_prompt = gr.Textbox(
                    label="Prompt", value="Change the style to cinematic anime", lines=3
                )
                v2v_button = gr.Button("Generate Video-to-Video", variant="primary")

            duration_input = gr.Slider(
                label="Video Duration (seconds)",
                minimum=0.3,
                maximum=8.5,
                value=2,
                step=0.1,
                info="Target video duration (0.3s to 8.5s)",
            )

            with gr.Accordion("⚡ Configuración Inteligente ZeroGPU", open=False):
                gr.Markdown("""
                **Optimiza automáticamente los parámetros para maximizar la duración del video en ZeroGPU.**
                
                ZeroGPU tiene tiempo limitado de GPU. Esta configuración ajusta resolución y multi-scale 
                para permitir videos más largos (hasta 10 segundos) sin errores de timeout.
                """)

                smart_config_enable = gr.Checkbox(
                    label="Activar Configuración Inteligente",
                    value=False,
                    info="Ajusta automáticamente resolución y multi-scale según la duración seleccionada",
                )

                smart_target_duration = gr.Slider(
                    label="Duración Objetivo (segundos)",
                    minimum=2,
                    maximum=10,
                    value=5,
                    step=0.5,
                    visible=False,
                    info="La configuración se optimizará para lograr esta duración",
                )

                gr.Markdown(
                    """
                **Recomendaciones por duración (optimizadas para ZeroGPU 70GB):**
                - **2-3s**: Resolución alta (640-704×832-960), multi-scale → Máxima calidad
                - **3-4s**: Resolución media-alta (576×768), multi-scale → Alta calidad
                - **4-5s**: Resolución media (480×640), single-pass → Balance
                - **5-6s**: Resolución media-baja (480×640), single-pass → Duración media
                - **6-7s**: Resolución baja (416×544), single-pass → Duración larga
                - **8-10s**: Resolución muy baja (384×512), single-pass → Máxima duración
                
                ⚠️ **Importante**: Duraciones >7s usan resolución muy baja para evitar timeout.
                """,
                    visible=False,
                    elem_id="smart_recommendations",
                )

            improve_texture = gr.Checkbox(
                label="Improve Texture (multi-scale)",
                value=True,
                visible=False,
                info="Uses a two-pass generation for better quality, but is slower. Recommended for final output.",
            )
            
            slow_motion_control = gr.Checkbox(
                label="🎬 Slow Motion",
                value=False,
                info="✅ Activado = Slow motion cinematográfico | ❌ Desactivado = Velocidad normal/real-time (recomendado)",
            )

        with gr.Column(scale=1):
            output_video = gr.Video(label="Your Video", interactive=False)

    with gr.Accordion("Advanced Configurations", open=False):
        mode = gr.Dropdown(
            ["text-to-video", "image-to-video", "video-to-video"],
            label="task",
            value="image-to-video",
            visible=False,
        )
        negative_prompt_input = gr.Textbox(
            label="Negative Prompt",
            value="worst quality, inconsistent motion, blurry, jittery, distorted",
            lines=2,
        )
        with gr.Row():
            seed_input = gr.Number(
                label="Seed", value=42, precision=0, minimum=0, maximum=2**32 - 1
            )
            randomize_seed_input = gr.Checkbox(label="Randomize Seed", value=True)
        with gr.Row(visible=False):
            guidance_scale_input = gr.Slider(
                label="Guidance Scale (CFG)",
                minimum=1.0,
                maximum=10.0,
                value=PIPELINE_CONFIG_YAML.get("first_pass", {}).get(
                    "guidance_scale", 1.0
                ),
                step=0.1,
                info="Controls how much the prompt influences the output. Higher values = stronger influence.",
            )
        with gr.Row():
            height_input = gr.Slider(
                label="Height",
                value=512,
                step=32,
                minimum=MIN_DIM_SLIDER,
                maximum=MAX_IMAGE_SIZE,
                info="Must be divisible by 32.",
            )
            width_input = gr.Slider(
                label="Width",
                value=704,
                step=32,
                minimum=MIN_DIM_SLIDER,
                maximum=MAX_IMAGE_SIZE,
                info="Must be divisible by 32.",
            )

    def apply_smart_config(enable, target_duration):
        """Aplica configuración inteligente para optimizar tiempo de GPU"""
        if not enable:
            return (
                gr.update(),  # duration_input
                gr.update(),  # height_input
                gr.update(),  # width_input
                gr.update(),  # improve_texture
                gr.update(visible=False),  # smart_target_duration
            )

        # Configuración optimizada por duración con límites estrictos para ZeroGPU
        if target_duration <= 2.5:
            # 2-2.5s: Máxima calidad (60 frames @ 30fps)
            height, width = 704, 960
            multi_scale = True
            duration = 2.0
        elif target_duration <= 3.5:
            # 3-3.5s: Alta calidad (90 frames @ 30fps)
            height, width = 640, 832
            multi_scale = True
            duration = 3.0
        elif target_duration <= 4.5:
            # 4-4.5s: Balance (120 frames @ 30fps)
            height, width = 576, 768
            multi_scale = False
            duration = 4.0
        elif target_duration <= 6.0:
            # 5-6s: Duración media (150 frames @ 30fps)
            height, width = 480, 640
            multi_scale = False
            duration = 5.0
        elif target_duration <= 7.5:
            # 6.5-7.5s: Duración larga (190 frames @ 30fps)
            height, width = 416, 544
            multi_scale = False
            duration = 6.3  # ~189 frames
        else:
            # 8-10s: Máxima duración con límite estricto
            # LÍMITE: 8.5s = 257 frames es el máximo absoluto del modelo
            # Pero con resolución MUY baja para caber en 70GB VRAM
            height, width = 384, 512
            multi_scale = False
            duration = 8.5  # 257 frames (máximo del modelo)

        return (
            gr.update(value=duration),  # duration_input
            gr.update(value=height),  # height_input
            gr.update(value=width),  # width_input
            gr.update(value=multi_scale),  # improve_texture
            gr.update(visible=True),  # smart_target_duration
        )

    def handle_image_upload_for_dims(image_filepath, current_h, current_w):
        if not image_filepath:
            return gr.update(value=current_h), gr.update(value=current_w)
        try:
            img = Image.open(image_filepath)
            orig_w, orig_h = img.size
            new_h, new_w = calculate_new_dimensions(orig_w, orig_h)
            return gr.update(value=new_h), gr.update(value=new_w)
        except Exception as e:
            print(f"Error processing image for dimension update: {e}")
            return gr.update(value=current_h), gr.update(value=current_w)

    def handle_video_upload_for_dims(video_filepath, current_h, current_w):
        if not video_filepath:
            return gr.update(value=current_h), gr.update(value=current_w)
        try:
            video_filepath_str = str(video_filepath)
            if not os.path.exists(video_filepath_str):
                print(
                    f"Video file path does not exist for dimension update: {video_filepath_str}"
                )
                return gr.update(value=current_h), gr.update(value=current_w)

            orig_w, orig_h = -1, -1
            with imageio.get_reader(video_filepath_str) as reader:
                meta = reader.get_meta_data()
                if "size" in meta:
                    orig_w, orig_h = meta["size"]
                else:
                    try:
                        first_frame = reader.get_data(0)
                        orig_h, orig_w = first_frame.shape[0], first_frame.shape[1]
                    except Exception as e_frame:
                        print(
                            f"Could not get video size from metadata or first frame: {e_frame}"
                        )
                        return gr.update(value=current_h), gr.update(value=current_w)

            if orig_w == -1 or orig_h == -1:
                print(f"Could not determine dimensions for video: {video_filepath_str}")
                return gr.update(value=current_h), gr.update(value=current_w)

            new_h, new_w = calculate_new_dimensions(orig_w, orig_h)
            return gr.update(value=new_h), gr.update(value=new_w)
        except Exception as e:
            print(
                f"Error processing video for dimension update: {e} (Path: {video_filepath}, Type: {type(video_filepath)})"
            )
            return gr.update(value=current_h), gr.update(value=current_w)

    image_i2v.upload(
        fn=handle_image_upload_for_dims,
        inputs=[image_i2v, height_input, width_input],
        outputs=[height_input, width_input],
    )
    video_v2v.upload(
        fn=handle_video_upload_for_dims,
        inputs=[video_v2v, height_input, width_input],
        outputs=[height_input, width_input],
    )

    # Event handlers para configuración inteligente
    smart_config_enable.change(
        fn=apply_smart_config,
        inputs=[smart_config_enable, smart_target_duration],
        outputs=[
            duration_input,
            height_input,
            width_input,
            improve_texture,
            smart_target_duration,
        ],
    )

    smart_target_duration.change(
        fn=apply_smart_config,
        inputs=[smart_config_enable, smart_target_duration],
        outputs=[
            duration_input,
            height_input,
            width_input,
            improve_texture,
            smart_target_duration,
        ],
    )

    image_tab.select(fn=update_task_image, outputs=[mode])
    text_tab.select(fn=update_task_text, outputs=[mode])

    t2v_inputs = [
        t2v_prompt,
        negative_prompt_input,
        image_n_hidden,
        video_n_hidden,
        height_input,
        width_input,
        mode,
        duration_input,
        frames_to_use,
        seed_input,
        randomize_seed_input,
        guidance_scale_input,
        improve_texture,
        slow_motion_control,
    ]

    i2v_inputs = [
        i2v_prompt,
        negative_prompt_input,
        image_i2v,
        video_i_hidden,
        height_input,
        width_input,
        mode,
        duration_input,
        frames_to_use,
        seed_input,
        randomize_seed_input,
        guidance_scale_input,
        improve_texture,
        slow_motion_control,
    ]

    v2v_inputs = [
        v2v_prompt,
        negative_prompt_input,
        image_v_hidden,
        video_v2v,
        height_input,
        width_input,
        mode,
        duration_input,
        frames_to_use,
        seed_input,
        randomize_seed_input,
        guidance_scale_input,
        improve_texture,
        slow_motion_control,
    ]

    t2v_button.click(
        fn=generate,
        inputs=t2v_inputs,
        outputs=[output_video, seed_input],
        api_name="text_to_video",
    )
    i2v_button.click(
        fn=generate,
        inputs=i2v_inputs,
        outputs=[output_video, seed_input],
        api_name="image_to_video",
    )
    v2v_button.click(
        fn=generate,
        inputs=v2v_inputs,
        outputs=[output_video, seed_input],
        api_name="video_to_video",
    )
    
    # Footer with links
    gr.Markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; opacity: 0.8;">
        <p>
            <a href="https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.8-distilled.safetensors" target="_blank">Model</a> | 
            <a href="https://github.com/Lightricks/LTX-Video" target="_blank">GitHub</a> | 
            <a href="https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled#diffusers-🧨" target="_blank">Diffusers 🧨</a>
        </p>
    </div>
    """)

if __name__ == "__main__":
    if os.path.exists(models_dir) and os.path.isdir(models_dir):
        print(f"Model directory: {Path(models_dir).resolve()}")

    demo.queue().launch(debug=True, share=False, mcp_server=True)
