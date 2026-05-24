# 🌤️ Seasons of Code 2026: Diffusion Models from Scratch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-latest-red)
![Status](https://img.shields.io/badge/Status-Active-success)
![Duration](https://img.shields.io/badge/Duration-8_Weeks-orange)

Welcome to the **Seasons of Code 2026** track for Generative AI! 

In this 8-week mentorship program, you will build—entirely from scratch—a working diffusion model capable of generating original images. There are no black-box API calls here. By the end of this journey, you will have implemented and deeply understood every component of a modern diffusion model, from the UNet architecture to the noise scheduler and the sampling loop.

---

## 🎯 Program Overview

**Cohort Size:** 6 mentees  
**Time Commitment:** 8–12 hours/week  
**Prerequisites:** Python proficiency, basic Machine Learning (sklearn-level), and comfort with NumPy and linear algebra.

### What You Will Build & Deliver
By Week 8, each mentee will have produced a complete portfolio project:
1. **Clean Codebase:** A documented, public GitHub repository.
2. **Trained Model:** A diffusion model trained on a custom dataset.
3. **Interactive Demo:** A Gradio application deployed on Hugging Face Spaces.
4. **Technical Blog Post:** A detailed write-up of your implementation and findings.
5. **Final Showcase:** A 5-minute final presentation video.

### Core Technologies & Concepts
*   **PyTorch Fluency:** Tensors, autograd, custom training loops, and GPU optimization.
*   **Architecture:** Convolutional Neural Networks and the UNet design pattern.
*   **Mathematics:** The mechanics of Denoising Diffusion Probabilistic Models (DDPM).
*   **Advanced Sampling:** Deterministic sampling (DDIM) and Classifier-Free Guidance (CFG).

---

## 📅 Weekly Curriculum

### Week 0: Pre-Program Setup
*   **Goal:** Environment configuration and baseline alignment.
*   **Tasks:** Set up GitHub, Google Colab/Kaggle, and Hugging Face accounts. Implement a simple linear regression with gradient descent in NumPy. Review 3Blue1Brown's neural network series.

### Week 1: PyTorch Foundations
*   **Goal:** Transition from `sklearn` to PyTorch.
*   **Tasks:** Understand the PyTorch computational graph. Build a custom `Dataset` class and a manual training loop. Train a feedforward network on MNIST to >97% accuracy.
*   > **Checkpoint:** Walk through what `loss.backward()` does internally. What gets modified, and where do gradients live?

### Week 2: Convolutional Networks & The UNet
*   **Goal:** Build the architecture that powers diffusion models.
*   **Tasks:** Implement modular `DoubleConv`, `Down`, and `Up` blocks. Construct a full UNet from scratch and train it as a denoising autoencoder on corrupted images.
*   > **Checkpoint:** Why are skip connections in a UNet important? What happens to information flow without them?

### Week 3: The Forward Diffusion Process
*   **Goal:** Systematically destroy an image with noise.
*   **Tasks:** Implement linear and cosine noise schedules. Write the closed-form $q(x_t \mid x_0)$ sampling using the reparameterization trick. Build the dataset pipeline to produce `(noisy_image, timestep, noise)` tuples.
*   > **Checkpoint:** Explain why we can sample $x_t$ directly from $x_0$ in one step instead of iterating $t$ times.

### Week 4: Training the Reverse Process
*   **Goal:** Generate your first images.
*   **Tasks:** Inject sinusoidal timestep embeddings into the UNet. Train a full DDPM on MNIST/Fashion-MNIST. Implement the iterative sampling loop to denoise from pure Gaussian noise.
*   **Milestone:** Mid-Program Demo Day.

### Week 5: Faster Sampling with DDIM
*   **Goal:** Reduce sampling steps from 1000 to 50 without quality loss.
*   **Tasks:** Understand stochastic vs. deterministic sampling. Implement a `DDIMScheduler` with a configurable $\eta$ parameter. Benchmark speed and quality tradeoffs.
*   > **Checkpoint:** What is the key insight that allows DDIM to use the same trained model as DDPM but generate samples faster?

### Week 6: Conditional Generation
*   **Goal:** Direct the model to generate specific classes.
*   **Tasks:** Implement class label embeddings. Add random label dropout during training (10–20%). Implement Classifier-Free Guidance (CFG) sampling with an adjustable guidance scale $w$.

### Week 7: Custom Dataset & Final Run
*   **Goal:** Apply your knowledge to a unique domain.
*   **Tasks:** Curate, preprocess, and augment a custom dataset (e.g., Pokemon sprites, retro logos, textile patterns). Execute a full training run (>100 epochs). Push the final `model.pt` checkpoint to the Hugging Face Hub.

### Week 8: Demo, Documentation, & Showcase
*   **Goal:** Ship it. Show it. Explain it.
*   **Tasks:** Wrap your model in an interactive Gradio app. Deploy to Hugging Face Spaces. Publish a 1500+ word technical blog post. Present at the Final Demo Day.

---

## 🛠️ Repository Structure

Use this structure as the template for your project repository:

```text
diffusion-soc-2026/
├── README.md               # Project overview, setup, and results
├── requirements.txt        # Python dependencies
├── data/                   # (Gitignored) Raw and processed datasets
├── src/                    # Source code module
│   ├── __init__.py
│   ├── dataset.py          # Data loaders and preprocessing
│   ├── model.py            # UNet architecture and embeddings
│   ├── scheduler.py        # DDPM/DDIM noise schedulers
│   ├── train.py            # Training loops and logging
│   └── sample.py           # Generation scripts
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                # Utility scripts (e.g., download_data.py)
├── samples/                # Generated image grids 
└── checkpoints/            # (Gitignored) Model weights (.pt files)
