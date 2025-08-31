# Enhanced Iris-Based Authentication System

This project implements a robust iris-based authentication system utilizing advanced deep learning techniques and homomorphic encryption. The system ensures secure and reliable identity verification while maintaining data privacy through encryption.

## Features

- **Iris Segmentation**: Uses Canny Edge Detection and Hough Circle Transform for precise iris segmentation.
- **Data Enhancement**: Implements preprocessing steps for improved data quality, including normalization and enhancement.
- **Feature Extraction**: Extracts unique features from the iris data to ensure high accuracy in authentication.
- **Homomorphic Encryption**: Ensures privacy-preserving computation by processing encrypted iris features.
- **Homomorphic Encryption Key Generation**: Produces secure encryption keys based on extracted and encrypted iris features.
- **Error Correction**: Handles inaccuracies in feature extraction with robust error correction techniques.

## Workflow

1. **Dataset Preparation**:
   - Obtain a suitable iris dataset.
   - Annotate and preprocess the data for analysis.

2. **Iris Segmentation**:
   - Employ Label studio for accurate iris segmentation.

3. **Normalization & Enhancement**:
   - Normalize and enhance iris images for uniform feature extraction.

4. **Feature Extraction**:
   - Extract distinguishing features using advanced algorithms.

5. **Homomorphic Encryption**:
   - Encrypt the extracted features using a homomorphic encryption library (e.g., PySEAL, TenSEAL).

6. **Error Correction**:
   - Apply error correction to ensure accurate encryption key generation.

7. **Encryption Key Generation**:
   - Generate encryption keys from the processed and encrypted iris data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amal957/eyeiris-authentication
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

- Python 3.8+
- TensorFlow/PyTorch
- OpenCV
- PySEAL/TenSEAL (or other homomorphic encryption libraries)
- NumPy
- Pandas

## Results

- High accuracy in segmentation and feature extraction.
- Strong encryption key generation ensuring secure and privacy-preserving authentication.

## Future Enhancements

- Extend support to multi-modal biometrics.
- Optimize the system for real-time applications.
- Explore alternative homomorphic encryption schemes for improved efficiency.

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue to improve the project.

## Acknowledgments

- Inspiration from iris recognition and homomorphic encryption research.
- Support from the deep learning and cryptographic communities.
