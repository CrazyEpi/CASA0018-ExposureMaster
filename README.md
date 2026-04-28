# CASA0018-ExposureMaster

dataset (20GB): https://drive.google.com/drive/folders/1Zc16SMJb0zgco7fLR3mlmqkdFz2pC9Sd?usp=drive_link

Exposure Expert is a mobile-based computer vision system designed to provide real-time exposure guidance for analog photography. By utilizing a fine-tuned MobileNetV2 architecture, the system performs local inference on ambient lighting conditions to predict optimal Exposure Values (EV), mitigating film waste through haptic and visual feedback.

## Technical Architecture

### 1. Data Engineering and Simulation Pipeline
The model was trained on a composite dataset of 8,407 images, utilizing three distinct sources to ensure environmental and lighting diversity.

* **SICE Dataset Integration:** Leveraged multi-exposure sequences where the median frame serves as the ground truth (0 EV) for normal exposure.
* **Linear RGB Exposure Simulation:** Custom Python scripts processed 16-bit Nikon RAW (NEF) data. Exposure shifts were mathematically simulated in a linear domain before applying Gamma 2.2 correction:
    $$I_{shifted} = I_{linear} \times 2^{EV_{shift}}$$
* **Portrait Domain Adaptation:** To improve performance on human subjects, a subset of the WIDER Face dataset was processed. Since these were non-linear JPEGs, inverse gamma decoding was required to approximate linear space for EV transformation:
    $$I_{shifted} = (I_{jpeg}^{2.2} \times 2^{EV_{shift}})^{1/2.2}$$

### 2. Model Development and Training Strategy
The system utilizes **MobileNetV2** as the backbone for its efficient use of inverted residuals and linear bottlenecks, optimized for mobile ARM processors.

* **Training Regime:** A two-stage transfer learning approach.
    * **Stage 1:** Classification head training with frozen backbone (70 epochs, Cosine Decay scheduler).
    * **Stage 2:** Fine-tuning of the top 20 layers (50 epochs, learning rate $1 \times 10^{-6}$).
* **Regularization:** Implementation of a 0.4 Dropout rate to prevent overfitting on the custom dataset.
* **Stability Measures:** All Batch Normalization (BN) layers remained locked during Stage 2 fine-tuning to prevent internal covariate shift corruption from the smaller domain-specific dataset.
* **Class Imbalance:** Applied class weighting to the loss function to compensate for the higher frequency of "Normal" exposure samples.

### 3. Edge Deployment and Optimization
The inference engine is integrated into a Flutter-based mobile application, focusing on low-latency execution.

* **Quantization:** The TensorFlow Lite model underwent weight quantization, reducing the binary footprint to ~4MB and significantly decreasing CPU cycles per inference.
* **Haptic Feedback Protocol:**
    * **Normal (Success):** Single short vibration pulse.
    * **Under/Over (Warning):** Heavy double-pulse vibration.
* **Hardware Abstraction:** Due to API restrictions on direct ISO/Shutter control in cross-platform frameworks, a software-based exposure compensation layer was implemented to simulate hardware sensitivity adjustments.

## Performance Evaluation

### Accuracy Metrics
The model achieved a final validation accuracy of **79.1%**. A notable observation during training was that validation accuracy consistently tracked higher than training accuracy, indicating that the augmentation and dropout layers successfully generalized the model's predictive capabilities.

### Confusion Matrix Analysis
Experimental results on the test set indicate:
* **Zero Fatal Errors:** The model never misclassifies extreme underexposure as overexposure or vice-versa.
* **Bias Optimization:** Misclassifications typically skew toward "Overexposed" rather than "Underexposed." In the context of film chemistry (specifically C-41 or instant film), this bias is beneficial as film maintains higher latitude in highlights than in shadows.

## Installation and Usage
1.  Clone the repository: `git clone https://github.com/CrazyEpi/CASA0018-ExposureMaster.git`
2.  Install Flutter dependencies: `flutter pub get`
3.  Deploy to an Android device with camera and vibrator permissions enabled.
4.  The `.tflite` model is located in the `assets/` directory.
