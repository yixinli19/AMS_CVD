# AMS_CVD: Adaptive Model Selection for Cardiovascular Disease Detection

AMS_CVD is an advanced system designed to detect cardiovascular diseases (CVD) in real-time, tailored for deployment on embedded systems such as the Raspberry Pi 4. The system employs a novel Convolutional Neural Network (CNN) architecture enhanced with Residual Blocks and a Global Attention mechanism using Squeeze-and-Excitation (SE) layers. This design ensures high diagnostic performance while maintaining computational efficiency.

## Features

1. Adaptive Model Selection (AMS): Dynamically adjusts model complexity based on real-time heart rate, optimizing performance across varying physiological conditions.
 
2. Anytime Model with Parameter Sharing: Integrates multiple model complexities into a single, parameter-efficient network, facilitating seamless transitions between complexity levels and reducing memory requirements.
 
3. Performance Optimization: Evaluates diagnostic accuracy using varying numbers of heartbeats as input, balancing accuracy and computational efficiency.
 
4. Embedded System Compatibility: Extensively tested on Raspberry Pi 4 using real-world ECG datasets, demonstrating high diagnostic accuracy while adhering to real-time processing constraints.

## Dataset

The model utilizes datasets from the PhysioNet 2021 Challenge and the MIT-BIH Arrhythmia Database.
PhysioNet 2021 Challenge: A comprehensive collection of ECG recordings used for training and evaluation.
Link: https://moody-challenge.physionet.org/2021/


## References
Reyna, M. A., et al. (2021). Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021.
