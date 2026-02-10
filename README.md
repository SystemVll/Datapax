> [!NOTE]
> Datapax is not production-ready and is currently in active development.</br> 
> It achieve **79.5%** success rate on a small test set of 1000 images.

<div align="center">

<img width="256" height="256" alt="Datapax Logo" src="https://github.com/user-attachments/assets/f1a9a15a-8104-4ef8-bf0c-2803a569e4e2" />

# Datapax

**AI-powered dataset patching and normalization pipeline for image data.**

Intelligently normalize images of any size and aspect ratio into fixed resolutions<br />
using AI-assisted outpainting not naive resizing or cropping.

---

![Status](https://img.shields.io/badge/Status-Prototype-orange) ![Backend](https://img.shields.io/badge/Backend-Qwen%20Image%20Edit%20Plus-blue) ![Framework](https://img.shields.io/badge/Framework-PyTorch-red)

---

</div>

The project currently uses **Qwen Image Edit Plus**, but the architecture is designed to be model-agnostic and replaceable.

## 🎬 Example

<p align="center">
  <img src="https://github.com/user-attachments/assets/9def1e56-1fac-4155-92f4-8abcda5c04a7" width="360" />
  <img src="https://github.com/user-attachments/assets/14764352-0eb1-434b-ad83-92a3befd61d1" width="360" />
</p>

#### What happened here?

- The original image of a **Sukhoi-57** aircraft had a resolution of **1500×1000**
- The target dataset resolution was **720×720**
- Instead of cropping or stretching the image:
  - Datapax **kept the full aircraft visible**
  - Preserved **scale, proportions, lighting, and perspective**
  - **Outpainted missing pixels** to fill the square frame naturally
- The background was extended using AI, without introducing new objects or stylistic changes

> This approach produces dataset-ready images while avoiding the common pitfalls of traditional resizing pipelines.

---

## 🎯 Why Datapax?

**What Problem Does Datapax Solve?**

Traditional dataset preprocessing often relies on:
- Center crops
- Resizing with distortion
- Manual padding
- Loss of important visual context

Datapax aims to:
- Preserve the **entire subject**
- Maintain **original proportions**
- Keep **background, lighting, and perspective intact**
- Use AI-assisted outpainting and editing to fill missing areas naturally

**Use Cases:**
- Vision model training
- Diffusion datasets
- Image-to-image and multimodal models
- Any workflow that needs clean, consistent image sizes without destroying content

---

## ✨ Core Features

- **AI-based image normalization** (e.g. random size → `512×512`)
- **Intelligent outpainting** instead of cropping
- **Subject-aware framing**
- **Preserves colors, lighting, and sharpness**
- **Designed for dataset-scale processing**
- **Model-agnostic pipeline** (Qwen is just the first backend)

---

## 🔧 Current Backend

- **Image Editing Model:** Qwen Image Edit Plus  
- **Framework:** PyTorch

The model choice is **not hardcoded** and will be swappable in future versions.

---

## 💻 Environment

Tested with:

- **PyTorch:** `2.10.0+cu128`
- **CUDA:** 12.8
- **OS:** Windows & Linux

---

## 📊 Project Status

Datapax is currently in **active prototyping**.

Planned milestones:
1. Working end-to-end example
2. Reproducible dataset patching pipeline
3. Documentation & configuration cleanup
4. Open-source release

Once milestone **#2** is reached, the repository will be made public immediately.

---

## 🗺️ Roadmap (Planned)

- [ ] Modular backend interface (multiple image-edit models)
- [ ] CLI interface for dataset processing
- [x] Batch processing
- [ ] Metadata & annotation preservation
- [ ] Config-driven pipelines
- [x] Open-source release

---

## 📜 License

**TBD**  
The license will be defined at the time of the open-source release.

---

## 📝 Notes

This project is experimental by nature. APIs, behavior, and internal structure may change rapidly until a stable release is published.

Feedback and ideas are welcome once the repository opens.

---

<p align="center">
  <b>Built with ❤️ for the AI community</b><br>
  <i>Making dataset preparation accessible and intelligent</i>
</p>
