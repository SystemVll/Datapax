# ANSI color codes
class Colors:
    ORANGE = "\033[38;5;214m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

print(f"""{Colors.ORANGE}{Colors.BOLD}
██████╗  █████╗ ████████╗ █████╗ ██████╗  █████╗ ██╗  ██╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗╚██╗██╔╝
██║  ██║███████║   ██║   ███████║██████╔╝███████║ ╚███╔╝ 
██║  ██║██╔══██║   ██║   ██╔══██║██╔═══╝ ██╔══██║ ██╔██╗ 
██████╔╝██║  ██║   ██║   ██║  ██║██║     ██║  ██║██╔╝ ██╗
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝
{Colors.RESET}""")

import sys
import os
import gc
import random


# Add ComfyUI to path - adjust this path to your ComfyUI installation
COMFYUI_PATH = r"C:\Users\User\Documents\ComfyUI\ComfyUI"
if os.path.exists(COMFYUI_PATH):
    sys.path.insert(0, COMFYUI_PATH)

import torch
from PIL import Image
import math
import numpy as np

# Configuration
MODEL_PATH = "models/Qwen-Rapid-AIO-SFW-v23.safetensors"
INPUT_DIR = "data/inputs"
OUTPUT_DIR = "data/outputs"
PROMPT = "Seamlessly outpaint the image while keeping the entire plane fully visible, centered, and in correct real-world proportions; preserve the original background, lighting, colors, sharpness, texture, and perspective exactly as-is without any alteration; fill missing or extended areas naturally using only the existing background and visual context; add pixels only where necessary for completion with no removal, replacement, or modification of existing pixels; maintain strict photorealism with a neutral, faithful reconstruction, dataset-safe output, and no artistic interpretation or enhancements; negative: text, logos, banners, watermarks, captions, borders, cropping, cut-off subject, censorship bars, blur, distortion, artifacts, compression noise, PNG transparency, added objects, removed details, style change, stylized look, cinematic lighting, dramatic shadows, illustration, painting, fantasy, surrealism."
WIDTH = 720
HEIGHT = 720
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 1.0
SAMPLER = "sa_solver"
SCHEDULER = "beta"

# Create directories if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get list of input images
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
if not input_files:
    print(f"{Colors.YELLOW}⚠ No input images found in {INPUT_DIR}{Colors.RESET}")
    print(f"{Colors.GRAY}  Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}{Colors.RESET}")
    sys.exit(0)

print(f"{Colors.CYAN}📁 Found {len(input_files)} image(s) to process{Colors.RESET}")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"{Colors.CYAN}⚡ Using device: {Colors.BOLD}{device}{Colors.RESET}")

# Clear CUDA cache at startup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# This simulates running ComfyUI with --lowvram flag
sys.argv = [sys.argv[0], "--lowvram"]

try:
    # Import ComfyUI modules
    import comfy.sd
    import comfy.utils
    import comfy.model_management
    import comfy.samplers
    import nodes

    print(f"{Colors.GREEN}✓ ComfyUI modules loaded successfully{Colors.RESET}")

    with torch.inference_mode():
        # Load the checkpoint
        print(f"\n{Colors.CYAN}╭─ Loading Checkpoint{Colors.RESET}")
        print(f"{Colors.CYAN}│  {Colors.GRAY}{MODEL_PATH}{Colors.RESET}")
        ckpt_path = MODEL_PATH
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.abspath(ckpt_path)

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=None
        )
        model, clip, vae = out[:3]
        print(f"{Colors.CYAN}╰─ {Colors.GREEN}✓ Model, CLIP, VAE loaded{Colors.RESET}")

        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

        # Process each input file
        for file_idx, input_filename in enumerate(input_files, 1):
            INPUT_IMAGE_PATH = os.path.join(INPUT_DIR, input_filename)
            base_name = os.path.splitext(input_filename)[0]
            OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, f"{base_name}_output.png")
            SEED = random.randint(0, 999999999)

            print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
            print(f"{Colors.CYAN}📷 Processing [{file_idx}/{len(input_files)}]: {Colors.BOLD}{input_filename}{Colors.RESET}")
            print(f"{'='*60}")

            # Load input image
            images = []
            ref_latents = []

            print(f"\n{Colors.CYAN}╭─ Loading Input Image{Colors.RESET}")
            print(f"{Colors.CYAN}│  {Colors.GRAY}{INPUT_IMAGE_PATH}{Colors.RESET}")
            input_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
            print(f"{Colors.CYAN}│  {Colors.GRAY}Original: {input_image.size[0]}x{input_image.size[1]}{Colors.RESET}")

            # Resize so shortest side = max(WIDTH, HEIGHT)
            target_short_side = max(WIDTH, HEIGHT)
            orig_w, orig_h = input_image.size
            if orig_w < orig_h:
                # Width is shorter
                new_w = target_short_side
                new_h = int(orig_h * (target_short_side / orig_w))
            else:
                # Height is shorter (or equal)
                new_h = target_short_side
                new_w = int(orig_w * (target_short_side / orig_h))
            input_image = input_image.resize((new_w, new_h), Image.LANCZOS)
            print(f"{Colors.CYAN}│  {Colors.GRAY}Resized:  {new_w}x{new_h} (shortest={target_short_side}){Colors.RESET}")

            # Convert to tensor (H, W, C) normalized 0-1
            img_array = np.array(input_image, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dim

            # Process for VL model (384x384 area)
            samples = img_tensor.movedim(-1, 1)
            total = 384 * 384
            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width_vl = round(samples.shape[3] * scale_by)
            height_vl = round(samples.shape[2] * scale_by)
            s = comfy.utils.common_upscale(samples, width_vl, height_vl, "area", "disabled")
            images.append(s.movedim(1, -1))

            # Process for VAE (1024x1024 area, divisible by 8)
            total = 1024 * 1024
            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width_vae = round(samples.shape[3] * scale_by / 8.0) * 8
            height_vae = round(samples.shape[2] * scale_by / 8.0) * 8
            s = comfy.utils.common_upscale(samples, width_vae, height_vae, "area", "disabled")
            ref_latent = vae.encode(s.movedim(1, -1)[:, :, :, :3])
            ref_latents.append(ref_latent)
            print(f"{Colors.CYAN}╰─ {Colors.GREEN}✓ Encoded to latent {Colors.GRAY}({ref_latent.shape[2]}x{ref_latent.shape[3]}x{ref_latent.shape[4]}){Colors.RESET}")

            image_prompt = ""
            for i in range(len(images)):
                image_prompt += f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"

            full_prompt = image_prompt + PROMPT
            print(f"\n{Colors.CYAN}╭─ Encoding Prompt{Colors.RESET}")
            print(f"{Colors.CYAN}│  {Colors.YELLOW}\"{PROMPT[:60]}{'...' if len(PROMPT) > 60 else ''}\"{Colors.RESET}")

            # Tokenize and encode
            tokens = clip.tokenize(full_prompt, images=images, llama_template=llama_template)
            conditioning = clip.encode_from_tokens_scheduled(tokens)

            # Add reference latents to conditioning
            if len(ref_latents) > 0:
                import node_helpers
                conditioning = node_helpers.conditioning_set_values(
                    conditioning,
                    {"reference_latents": ref_latents},
                    append=True
                )
            print(f"{Colors.CYAN}╰─ {Colors.GREEN}✓ Conditioning ready{Colors.RESET}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"\n{Colors.CYAN}╭─ Sampling{Colors.RESET}")
            print(f"{Colors.CYAN}│  {Colors.GRAY}Latent: {WIDTH}x{HEIGHT} │ Steps: {NUM_INFERENCE_STEPS} │ CFG: {GUIDANCE_SCALE}{Colors.RESET}")
            num_layers = len(images)
            latent = torch.zeros(
                [1, 16, num_layers + 1, HEIGHT // 8, WIDTH // 8],
                device=comfy.model_management.intermediate_device()
            )
            samples = {"samples": latent}

            neg_tokens = clip.tokenize("", llama_template=llama_template)
            negative = clip.encode_from_tokens_scheduled(neg_tokens)

            print(f"{Colors.CYAN}│  {Colors.GRAY}Sampler: {SAMPLER} │ Scheduler: {SCHEDULER}{Colors.RESET}")

            samples_out = nodes.common_ksampler(
                model=model,
                seed=SEED,
                steps=NUM_INFERENCE_STEPS,
                cfg=GUIDANCE_SCALE,
                sampler_name=SAMPLER,
                scheduler=SCHEDULER,
                positive=conditioning,
                negative=negative,
                latent=samples,
                denoise=1.0
            )

            print(f"{Colors.CYAN}╰─ {Colors.GREEN}✓ Sampling complete{Colors.RESET}")

            # Decode with VAE
            print(f"\n{Colors.CYAN}╭─ Decoding{Colors.RESET}")
            output_latent = samples_out[0]["samples"]
            print(f"{Colors.CYAN}│  {Colors.GRAY}Latent shape: {output_latent.shape}{Colors.RESET}")

            # Pass the full 5D latent - VAE.decode() handles extracting the first frame internally
            decoded = vae.decode(output_latent)
            print(f"{Colors.CYAN}│  {Colors.GRAY}Decoded shape: {decoded.shape}{Colors.RESET}")

            output_tensor = decoded

            if output_tensor.dim() == 5:
                output_tensor = output_tensor[0, 0]  # [H, W, C]
            elif output_tensor.dim() == 4:
                output_tensor = output_tensor[0]  # [H, W, C] assuming [B, H, W, C]

            output_tensor = output_tensor.squeeze()

            print(f"{Colors.CYAN}│  {Colors.GRAY}Final: {output_tensor.shape[0]}x{output_tensor.shape[1]}{Colors.RESET}")

            output_np = (output_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            output_np = np.ascontiguousarray(output_np)

            output_image = Image.fromarray(output_np)
            output_image.save(OUTPUT_IMAGE_PATH)
            print(f"{Colors.CYAN}╰─ {Colors.GREEN}✓ Saved to {Colors.BOLD}{OUTPUT_IMAGE_PATH}{Colors.RESET}")

        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All {len(input_files)} image(s) processed successfully!{Colors.RESET}")

except ImportError as e:
    print(f"\n{Colors.RED}✗ Could not import ComfyUI modules: {e}{Colors.RESET}")
    print(f"\n{Colors.YELLOW}To use this script, you need ComfyUI installed.{Colors.RESET}")
    print(f"{Colors.GRAY}Current COMFYUI_PATH: {COMFYUI_PATH}{Colors.RESET}")

except Exception as e:
    print(f"\n{Colors.RED}✗ Error: {e}{Colors.RESET}")
    import traceback
    traceback.print_exc()
