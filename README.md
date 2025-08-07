# Avalanche Risk Classification Pipeline

A comprehensive machine learning pipeline that classifies avalanche risk based on structured snowpack data and snow profile images using OpenCV feature extraction.

## 🎯 Overview

This pipeline combines:
1. **Structured CSV data** - 34 numerical features (grain size, shear strength, temperature, etc.)
2. **Snow profile images** - OpenCV-extracted features (histograms, edge detection, contours, texture)

**Target classes**: `Stable`, `Borderline`, `Unstable`

**Output**: Trained Random Forest Classifier saved as `avalanche_model.pkl`

## 🛠️ Features

### Modular Design
- **Data Loading & Preprocessing**: `load_and_preprocess_csv()`
- **OpenCV Feature Extraction**: `extract_opencv_features()`
- **Feature Fusion**: `combine_features()`
- **Model Training**: `train_model()`
- **Model Evaluation**: `evaluate_model()`

### OpenCV Features (20 per image)
- **Histogram** (16 features): Normalized pixel intensity distribution
- **Edge Detection**: Canny edge density
- **Contour Count**: Number of detected contours
- **Texture Metrics**: Mean and standard deviation of pixel intensities

### Machine Learning Pipeline
- **Random Forest Classifier** (100 estimators)
- **StandardScaler** for feature normalization
- **LabelEncoder** for target encoding
- **Comprehensive evaluation** with visualizations

## 📁 Project Structure

```
workspace/
├── avalanche_risk_classifier.py    # Main pipeline script
├── requirements.txt                 # Python dependencies
├── avalanche_model.pkl             # Trained model (output)
├── model_evaluation.png            # Evaluation plots (output)
├── sample_avalanche_data.csv       # Sample dataset
└── sample_images/                  # Sample snow profile images
    ├── profile_0000.jpg
    ├── profile_0001.jpg
    └── ...
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
python3 avalanche_risk_classifier.py
```

### 3. Expected Output
- Trained model: `avalanche_model.pkl`
- Evaluation plots: `model_evaluation.png`
- Console output with metrics and feature importances

## 📊 Pipeline Steps

### 1. Data Loading
- Loads CSV with 34 numerical features
- Handles missing values with median imputation
- Encodes target classes: Stable(0), Borderline(1), Unstable(2)

### 2. OpenCV Feature Extraction
- Processes snow profile images
- Extracts 20 features per image:
  - 16 histogram bins (normalized)
  - Edge density (Canny detection)
  - Contour count (normalized)
  - Mean intensity
  - Standard deviation of intensity

### 3. Feature Fusion
- Combines 34 structured features + 20 OpenCV features
- Results in 54-dimensional feature space

### 4. Model Training
- Random Forest with 100 trees
- 80/20 train-test split
- StandardScaler normalization
- Stratified sampling (with fallback)

### 5. Evaluation & Visualization
- Classification report (precision, recall, F1-score)
- Confusion matrix heatmap
- Top 10 feature importances
- Class distribution plots

## 📈 Sample Results

```
Classification Report:
              precision    recall  f1-score   support
  Borderline     0.31      0.27      0.29        67
      Stable     0.45      0.52      0.48        66
    Unstable     0.45      0.45      0.45        67

    accuracy                         0.41       200
```

## 🔧 Usage with Your Data

### CSV Format Required
```csv
grain_size,shear_strength,temperature,density,...,risk_level,profile_image
2.5,15.2,-5.1,0.4,...,Stable,images/profile_001.jpg
3.1,12.8,-3.2,0.5,...,Borderline,images/profile_002.jpg
```

### Using the Trained Model
```python
from avalanche_risk_classifier import AvalancheRiskClassifier
import joblib

# Load trained model
model_data = joblib.load('avalanche_model.pkl')
classifier = AvalancheRiskClassifier('data.csv', 'images/')
classifier.model = model_data['model']
classifier.scaler = model_data['scaler']
classifier.label_encoder = model_data['label_encoder']

# Make predictions on new data
# predictions = classifier.predict(new_features)
```

## 🎨 Customization

### Modify OpenCV Features
Edit the `extract_opencv_features()` method to add:
- Haralick texture features
- Local Binary Patterns (LBP)
- GLCM (Gray-Level Co-occurrence Matrix)
- Fourier descriptors

### Adjust Model Parameters
```python
self.model = RandomForestClassifier(
    n_estimators=200,      # More trees
    max_depth=15,          # Deeper trees
    min_samples_split=3,   # Different split criteria
    class_weight='balanced' # Handle class imbalance
)
```

## 📋 Requirements

- Python 3.7+
- OpenCV (cv2)
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib

## 🔍 Troubleshooting

### Missing Images
The pipeline gracefully handles missing images by using zero features. Warnings are displayed but don't stop execution.

### Class Imbalance
If stratified splitting fails due to insufficient samples in any class, the pipeline automatically falls back to random splitting.

### Memory Issues
For large datasets:
- Process images in batches
- Reduce image resolution
- Use feature selection techniques

## 📝 Notes

- Sample data is automatically generated for demonstration
- Real snow profile images would provide better OpenCV features
- The pipeline is designed to be modular and extensible
- All random seeds are set for reproducibility

## 🤝 Contributing

Feel free to enhance the pipeline by:
- Adding more sophisticated image features
- Implementing cross-validation
- Adding hyperparameter tuning
- Integrating deep learning models
