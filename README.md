# Deep Learning-based Image Colorization System 🎨

This repository contains a **GAN-based image colorization system**, designed to tackle common colorization issues such as **color bleeding** and **desaturation**, using a **classification-driven approach**.

---

## 🔬 Research Method

### Architecture
- Based on **Generative Adversarial Networks (GANs)**.
- Uses the **Pix2Pix** model for image-to-image translation.

### Color Space Selection
- Operates in **CIE Lab space**:
  - **L channel**: Luminance (Input)
  - **a and b channels**: Color information (Prediction targets)

### Learning Paradigm
A **two-step strategy**: **pre-training + adversarial fine-tuning**.

1. **Step 1 – Supervised Pre-training**
   - Generator is pre-trained using **Cross-Entropy (CE) loss**.
   - Ensures the model initially learns the color distribution.

2. **Step 2 – Adversarial Training**
   - Introduces a **discriminator**.
   - Enhances **color saturation** and **visual realism**.

### Loss Function Design
- **Generator Loss**:  
  L_G = L_GAN + λ * L_CE
  - \(L_{GAN}\): Binary Cross-Entropy (BCE) or Least Squares GAN (LSGAN) loss  
  - \(L_{CE}\): Cross Entropy Loss for 313 color categories

- **Discriminator Loss**:  
  - 1 for real images, 0 for fake images
  
- **Total Variation (TV) Loss** to reduce noise and unnatural color blocks

---

## 🎯 Aims and Outcomes

### Primary Aim
Develop a **classification-driven image colorization system** to reduce **color bleeding** and **desaturation** issues found in regression-based methods.

### Outcomes
- Source Code
- Evaluation Report & Data
- Functional Software(?????) Solution

---

## 🛠 Development of the Solution

### Practical Problem
- Address **desaturation**
- Efficiently process **large-scale image datasets** for colorization

### Software Architecture Design
- Modular design including:
  - Data Loading
  - Training
  - Validation

### Algorithmic Implementation
- **Pix2Pix** with **313-bin color classification**

### Tools & Technology Stack
- **Python**, **PyTorch**, **NumPy**, **Matplotlib**, **Scikit-image**, **FastAI (DynamicUnet)**

---

## ✅ Verification and Validation(?????)

### Verification
（through code????? or some real society services）
- like Range validation? (LAB values normalized correctly)

### Validation
- Randomly selected samples to evaluate generalization on unseen data(?????)

---

## ⚡ Critical Appraisal

### Potential Improvement
- Introduce **Attention Mechanism** to improve **global color consistency**(?????)

### Current Limitation
- Some colors may still lack authenticity in **complex semantic scenes**

---

## Question
- Training successfully transitioned from **RTX 3060** to **A6000**(?????)
- Validation still supported on **RTX 3060**

---

## 📂 File Structure

| File | Description |
|------|-------------|
| `load_lab_npy_data.py` | High-performance data loading using `mmap` |
| `train_pix2pix_from_npy.py` | Main GAN training logic and model definition |
| `val.py` | Inference and visualization script for model validation |

---

