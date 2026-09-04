
# 🌀 Denoising Diffusion Probabilistic Model (DDPM) From Scratch

A minimal implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** built from scratch using **PyTorch**.

This project implements the complete DDPM pipeline, from the forward diffusion process and noise prediction to reverse diffusion sampling and evaluation.

---

## 🚀 Overview

Diffusion models work by gradually adding Gaussian noise to data and training a neural network to reverse that process.

The pipeline implemented in this project is:

```text
Clean Image (x₀)
      ↓
Forward Diffusion
      ↓
Noisy Image (xₜ)
      ↓
Neural Network predicts noise
      ↓
Reverse Diffusion
      ↓
Generated Image (x₀)
```

---

## 🧠 Features

* 📈 Linear beta noise schedule
* 🔢 Alpha and cumulative alpha calculations
* 🌫️ Forward diffusion process
* 🎲 Random timestep sampling
* 🧮 Sinusoidal timestep embeddings
* 🧠 Tiny time-conditioned neural network
* 📉 Noise prediction using MSE loss
* 🔄 DDPM training loop
* 🎨 Reverse diffusion sampling
* 📊 Sample quality evaluation using nearest-neighbor MSE
* 🧪 Complete end-to-end DDPM experiment

---

## ⚙️ Implementation Pipeline

### 1. Forward Diffusion

Gaussian noise is gradually added to the original image:

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 +
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

where:

* \(x_0\) is the original image
* \(x_t\) is the noisy image
* \(\epsilon\) is Gaussian noise
* \(\bar{\alpha}_t\) controls the noise level

---

### 2. Noise Prediction

A time-conditioned neural network learns to predict the noise added at timestep \(t\).

The training objective is:

$$
L = \mathbb{E}\left[
\|\epsilon - \epsilon_\theta(x_t,t)\|^2
\right]
$$

---

### 3. Reverse Diffusion

Starting from pure Gaussian noise:

$$
x_T \sim \mathcal{N}(0,I)
$$

the model gradually removes noise:

```text
x_T → x_(T-1) → ... → x_1 → x_0
```

The final output is the generated sample.

---

## 📊 Results

The experiment was trained on a synthetic blob dataset.

| Metric                  |              Result |
| ----------------------- | ------------------: |
| Training Steps          |              **60** |
| Training Loss           | **1.0579 → 0.9380** |
| Pure Noise Baseline MSE |          **0.9739** |
| Trained Sample MSE      |          **0.6235** |
| Improvement             |          **0.3505** |

The trained DDPM samples achieved a lower MSE compared to pure random noise, indicating that the model learned useful structure from the dataset.

---

## 🖼️ Results Visualization

> Add the generated experiment image here after uploading it to your repository.

```markdown
![DDPM Results](assets/ddpm-results.png)
```

---

## 🛠️ Technologies Used

* Python
* PyTorch
* NumPy

---

## ▶️ Running the Experiment

Clone the repository:

[GitHub Repository](https://github.com/00-Abhishek/denoising-diffusion-ddpm-from-scratch?utm_source=chatgpt.com)

Install the required dependencies:

```bash
pip install torch numpy
```

Then run the implementation according to your project setup.

---

## 🎯 What I Learned

Building DDPM from scratch helped me understand:

* How noise schedules work
* The relationship between **β, α, and ᾱ**
* Forward and reverse diffusion processes
* Why diffusion models predict noise
* Time-conditioned neural networks
* Posterior mean and variance in DDPM sampling
* How generative models transform random noise into structured data

---

## 📚 References

* Ho et al., **Denoising Diffusion Probabilistic Models (2020)**
* PyTorch Documentation

---

## 👨‍💻 Author

**Abhishek**

🔗 [GitHub Profile](https://github.com/00-Abhishek?utm_source=chatgpt.com)

🔗 [Linkedin Profile](https://www.linkedin.com/in/abhishekpal-ai/).
   
---

⭐ If you found this project useful, consider giving the repository a star!

#MachineLearning #DeepLearning #PyTorch #DDPM #DiffusionModels #GenerativeAI
