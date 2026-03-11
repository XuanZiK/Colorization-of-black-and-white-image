Project: Deep Learning-based Image Colorization System

This repository contains the source code and documentation for a GAN-based image colorization system, specifically focusing on addressing color bleeding and desaturation issues using a classification-driven approach.
    Research Method

    Architecture: The project adopts a Generative Adversarial Networks (GAN) architecture, specifically the Pix2Pix model designed for image-to-image translation.

    Color Space Selection: Unlike the traditional RGB space, this project operates within the CIE Lab space.

        L channel: Luminance (Input).

        a and b channels: Color information (Prediction targets).

    Learning Paradigm: Adopt a two-step strategy of "pre-training + adversarial fine-tuning." * Step 1: The generator undergoes supervised pre-training using Cross-Entropy (CE) loss to ensure the model initially grasps color distribution.

        Step 2: An adversarial training process is introduced with a discriminator to enhance color saturation and visual realism.

    Loss Function Design:

        Total Generator Loss:
        LG​=LGAN​+λ⋅LCE​

            LGAN​: Utilize binary cross-entropy (BCE) or least squares (LSGAN) loss.

            LCE​: Cross Entropy Loss (Treats color prediction as a multi-class classification problem with 313 categories).

        Discriminator Loss: 1 for real images, 0 for fake images.

        Future Work: Total Variation (TV) Loss (Designed to reduce "noise" and "unnatural color blocks").

    Aims and Outcomes

    Primary Aim: To develop a deep learning-based image colorization system using a classification-driven approach to address the issue of color bleeding and desaturation commonly found in regression-based models.

    Outcomes:

        Source Code

        Evaluation Report & Data

        Software Solution

    Development of a Solution

Practical Problem: Solving desaturation and inefficient processing of large-scale image data for colorization.

    Software Architecture Design: (Modular design involving Data loading, Main Training, and Validation).

    Algorithmic Implementation: Implementation of Pix2Pix with 313-bin color classification.

    Tools and Technology Stack: Python, PyTorch, NumPy, Matplotlib, Skimage, FastAI (DynamicUnet).

    Verification and Validation

    Verification: Through code implementation checks, Smoke Tests on data loaders, and range validation (ensuring LAB values are normalized correctly).

    Validation:

        Randomly select samples to show generalization on unseen data.

        Visual comparison between Gray-scale input, Predicted result, and Ground Truth.

    Critical Appraisal of the Project

    Potential Improvement: Introducing Attention Mechanism to improve global color consistency.

    Current Limitation: Some colors may still lack total authenticity in complex semantic scenes.

Other question
    Resource Adaptation: Successfully transitioned training resources from RTX 3060 to A6000, while maintaining validation capabilities on the 3060.

    File Structure

    load_lab_npy_data.py: High-performance data loading using mmap.

    train_pix2pix_from_npy.py: Main GAN training logic and model definition.

    val.py: Inference and visualization script for model validation.
