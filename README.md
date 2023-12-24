# Fire_Smoke_Ddetection_openCV
to detect fire and smoke in videos using OpenCV library to be used in CCTV / Surveillance camera

# Smoke Detection using Gaussian Naive Bayes Classifier
This Python code is designed for smoke detection in images using a Gaussian Naive Bayes (GNB) classifier. The program extracts a variety of features, including color histograms, energy, FFT, local binary patterns (LBP), mean, variance, haze features, and contrast, from positive (smoke) and negative (non-smoke) image samples. These features are used to train a GNB classifier, which can then be employed to identify smoke in new images.

# Key Features:
Extraction of diverse image features
Training a Gaussian Naive Bayes classifier
Confusion matrix evaluation and visualization
# Usage:
Place positive (smoke) and negative (non-smoke) image samples in designated directories.
Run the script to extract features, train the classifier, and save the model.
Evaluate the model's performance using the confusion matrix.
# Dependencies:
NumPy
OpenCV
scikit-learn
PyWavelets
scikit-image
Matplotlib
