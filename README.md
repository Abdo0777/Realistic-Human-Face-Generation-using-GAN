# 🧑 Realistic Human Face Generation using GAN

A Deep Convolutional Generative Adversarial Network (DCGAN) built from scratch in TensorFlow/Keras to generate realistic synthetic human face images from random noise, trained on the CelebA dataset.

## 🎯 Objective

Generative Adversarial Networks learn to model a data distribution by pitting two neural networks against each other: a **Generator** that tries to create convincing fake images, and a **Discriminator** that tries to tell real images from generated ones. Through this adversarial process, the Generator progressively learns to produce increasingly realistic human faces from nothing but random noise vectors.

This project implements the full pipeline end-to-end:
- Data loading and preprocessing (cropping, resizing, normalization)
- Custom Generator and Discriminator architectures
- Adversarial training loop with loss monitoring
- Sample generation and quality tracking across training
- An interactive demo for generating new faces on demand

## 🏗️ Architecture

**Generator** — takes a 100-dimensional latent noise vector and upsamples it through a series of transposed convolution blocks (with BatchNorm + LeakyReLU) into a 64×64×3 RGB image, with a final `tanh` activation to output values in `[-1, 1]`.

**Discriminator** — takes a 64×64×3 image (real or generated) and downsamples it through convolutional blocks (with BatchNorm, LeakyReLU, and Dropout) into a single logit, classifying the image as real or fake.

Both networks use a deeper "double-conv-per-block" design (a stride-1 refinement convolution before each up/downsampling step) to give the model more representational capacity at each spatial resolution, improving output sharpness over a minimal single-conv-per-block baseline.

## 📊 Training Details

| | |
|---|---|
| Dataset | [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) (subset of 50,000 images) |
| Image size | 64×64×3 |
| Latent dimension | 100 |
| Batch size | 256 |
| Optimizer | Adam (lr=2e-4, β₁=0.5) for both networks |
| Loss | Binary cross-entropy (from logits) |
| Epochs | 150 |
| Precision | Mixed float16 (for faster GPU training) |
| Hardware | Kaggle GPU notebook |

Losses were monitored every epoch, with the Discriminator loss staying in a healthy 0.4–0.6 range throughout training — indicating a stable adversarial balance rather than either network collapsing or overpowering the other.

## 🖼️ Results

Sample faces generated from random noise, showing progression across training:

`epoch_001.png → epoch_050.png → epoch_150.png`

*(add your progression grid image and final face samples here)*

## 🚀 Demo

An interactive Gradio app is included (`app.py`) that loads the trained Generator and produces new random faces on demand, with adjustable batch size and optional seed control for reproducibility.

```bash
pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints in your terminal.

## 📁 Repository Structure

```
├── README.md
├── train_gan.ipynb        # Full training notebook
├── app.py                 # Gradio demo app
├── face_generator.keras   # Trained generator weights
├── requirements.txt
└── samples/                # Generated face samples across training
```

## 🛠️ Tech Stack

- **TensorFlow / Keras** — model building and training
- **Gradio** — interactive demo interface
- **NumPy / Matplotlib** — data handling and visualization
- **Kaggle GPU** — training environment

## 📌 Notes & Limitations

This is a from-scratch DCGAN, not a state-of-the-art architecture like StyleGAN — results show recognizable, varied faces with some artifacts (occasional texture noise, distorted features on failure cases), which is expected at this scale and training budget. The project's goal is to demonstrate a correct, working understanding of GAN architecture and adversarial training dynamics rather than to match production-grade face synthesis quality.

## 🙏 Acknowledgments

- Dataset: [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
