The main goal of this study is to improve the classification of tumors in mammogram images. Moreover, it explores the use of shape-based features to distinguish between benign and malignant tumors.
We propose a shape-driven approach based on fitted ellipses around tumor contours – introducing both an optimal ellipse fit and a horizontally-aligned fit as part of the analysis.


## Proposed solution - pipeline
The pipeline is designed to work with single abnormality mammograms from DDSM dataset (Digital Database for Screening Mammography).
The overall algorithm is structured as follows:
- Detecting the tumor region through preprocessing and segmentation
- Fitting an ellipse around the tumor contour (both optimal and horizontal)
- Extracting radial distance features
- Computing shape descriptors
-Classifying lesions to distinguish benign from malignant cases

## Technologies
Python, OpenCV, scikit-learn, Streamlit, NumPy, Matplotlib
