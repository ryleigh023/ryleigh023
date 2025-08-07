#!/usr/bin/env python3
"""
Avalanche Risk Classification Pipeline

A modular machine learning pipeline that classifies avalanche risk based on:
1. Structured CSV data (numerical features of snowpack)
2. Images of snow profiles (OpenCV feature extraction)

Author: AI Python Developer
"""

import os
import pandas as pd
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class AvalancheRiskClassifier:
    """
    A complete pipeline for avalanche risk classification using structured data and images.
    """
    
    def __init__(self, csv_path: str, image_dir: str = None):
        """
        Initialize the classifier with data paths.
        
        Args:
            csv_path: Path to the CSV file containing structured data
            image_dir: Directory containing profile images (optional, can be in CSV)
        """
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.feature_names = []
        self.is_fitted = False
        
    def load_and_preprocess_csv(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load CSV data and preprocess numerical features and target variable.
        
        Returns:
            Tuple of (features_df, target_encoded)
        """
        print("Loading and preprocessing CSV data...")
        
        # Load the CSV file
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        print(f"Loaded CSV with shape: {df.shape}")
        
        # Separate features and target
        if 'risk_level' not in df.columns:
            raise ValueError("Target column 'risk_level' not found in CSV")
        
        # Extract numerical features (excluding target and image path columns)
        exclude_cols = ['risk_level', 'profile_image']
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numerical_features if col not in exclude_cols]
        
        print(f"Found {len(feature_cols)} numerical features")
        
        # Handle missing values
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        features_df = df[feature_cols].copy()
        
        # Encode target variable
        target = self.label_encoder.fit_transform(df['risk_level'])
        print(f"Target classes: {self.label_encoder.classes_}")
        print(f"Target distribution: {np.bincount(target)}")
        
        return features_df, target
    
    def extract_opencv_features(self, image_path: str) -> np.ndarray:
        """
        Extract features from a snow profile image using OpenCV.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Array of extracted features
        """
        try:
            # Read image
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                return np.zeros(20)  # Return zero features if image missing
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image: {image_path}")
                return np.zeros(20)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Feature 1-16: Histogram of pixel intensities (16 bins)
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            hist_features = hist.flatten() / hist.sum()  # Normalize
            
            # Feature 17: Edge detection count using Canny
            edges = cv2.Canny(gray, 50, 150)
            edge_count = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            # Feature 18: Contour count
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_count = len(contours) / 1000.0  # Normalize
            
            # Feature 19: Mean intensity
            mean_intensity = np.mean(gray) / 255.0
            
            # Feature 20: Standard deviation of intensity
            std_intensity = np.std(gray) / 255.0
            
            # Combine all features
            opencv_features = np.concatenate([
                hist_features,
                [edge_count, contour_count, mean_intensity, std_intensity]
            ])
            
            return opencv_features
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.zeros(20)  # Return zero features on error
    
    def extract_features_from_all_images(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract OpenCV features from all images referenced in the dataframe.
        
        Args:
            df: DataFrame containing profile_image column
            
        Returns:
            Array of shape (n_samples, n_opencv_features)
        """
        print("Extracting OpenCV features from images...")
        
        if 'profile_image' not in df.columns:
            print("Warning: No 'profile_image' column found. Using zero features.")
            return np.zeros((len(df), 20))
        
        opencv_features_list = []
        
        for idx, image_path in enumerate(df['profile_image']):
            if pd.isna(image_path):
                opencv_features_list.append(np.zeros(20))
                continue
                
            # Handle relative paths
            if self.image_dir and not os.path.isabs(image_path):
                full_path = os.path.join(self.image_dir, image_path)
            else:
                full_path = image_path
                
            features = self.extract_opencv_features(full_path)
            opencv_features_list.append(features)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} images...")
        
        return np.array(opencv_features_list)
    
    def combine_features(self, structured_features: pd.DataFrame, opencv_features: np.ndarray) -> np.ndarray:
        """
        Combine structured CSV features with OpenCV features.
        
        Args:
            structured_features: DataFrame with numerical features
            opencv_features: Array with OpenCV features
            
        Returns:
            Combined feature array
        """
        print("Combining structured and image features...")
        
        # Store feature names for later use
        structured_names = list(structured_features.columns)
        opencv_names = [f'opencv_feature_{i+1}' for i in range(opencv_features.shape[1])]
        self.feature_names = structured_names + opencv_names
        
        # Combine features
        combined_features = np.hstack([
            structured_features.values,
            opencv_features
        ])
        
        print(f"Combined features shape: {combined_features.shape}")
        return combined_features
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the Random Forest model on combined features.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with training results
        """
        print("Training Random Forest model...")
        
        # Split data (handle potential class imbalance)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # Fall back to random split if stratification fails due to class imbalance
            print("Warning: Stratified split failed, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        results = {
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred,
            'accuracy': accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        print(f"Model trained successfully!")
        print(f"Training samples: {results['train_size']}")
        print(f"Test samples: {results['test_size']}")
        print(f"Test accuracy: {accuracy:.4f}")
        
        return results
    
    def evaluate_model(self, results: Dict[str, Any]) -> None:
        """
        Evaluate the model and create visualizations.
        
        Args:
            results: Dictionary containing training results
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        y_test = results['y_test']
        y_pred = results['y_pred']
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            digits=4
        ))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(15, 5))
        
        # Plot confusion matrix
        plt.subplot(1, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Plot feature importances (top 10)
        feature_importance = self.model.feature_importances_
        top_indices = np.argsort(feature_importance)[-10:]
        
        plt.subplot(1, 3, 2)
        plt.barh(range(10), feature_importance[top_indices])
        plt.yticks(range(10), [self.feature_names[i] for i in top_indices])
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Importance')
        
        # Plot class distribution
        plt.subplot(1, 3, 3)
        unique, counts = np.unique(y_test, return_counts=True)
        class_names = [self.label_encoder.classes_[i] for i in unique]
        plt.bar(class_names, counts)
        plt.title('Test Set Class Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('/workspace/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nEvaluation plots saved to: /workspace/model_evaluation.png")
    
    def save_model(self, filepath: str = '/workspace/avalanche_model.pkl') -> None:
        """
        Save the trained model and preprocessors.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved successfully to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a previously trained model.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.is_fitted = True
        print(f"Model loaded successfully from: {filepath}")


def create_sample_data():
    """
    Create sample data for testing the pipeline.
    """
    print("Creating sample dataset for demonstration...")
    
    # Create sample CSV data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    data = {
        'grain_size': np.random.normal(2.5, 0.8, n_samples),
        'shear_strength': np.random.normal(15.0, 5.0, n_samples),
        'temperature': np.random.normal(-5.0, 3.0, n_samples),
        'density': np.random.normal(0.4, 0.1, n_samples),
        'hardness': np.random.normal(3.0, 1.0, n_samples),
        'slope_angle': np.random.normal(35.0, 10.0, n_samples),
        'wind_speed': np.random.normal(10.0, 5.0, n_samples),
        'precipitation': np.random.normal(5.0, 3.0, n_samples),
        'depth': np.random.normal(150.0, 50.0, n_samples),
        'crystal_type': np.random.normal(2.0, 1.0, n_samples),
    }
    
    # Add more features to reach 34 total
    for i in range(10, 34):
        data[f'feature_{i+1}'] = np.random.normal(0, 1, n_samples)
    
    # Create risk levels based on some logic with balanced distribution
    risk_scores = (
        0.3 * data['shear_strength'] + 
        0.2 * data['slope_angle'] + 
        0.1 * data['wind_speed'] +
        0.1 * data['precipitation'] +
        np.random.normal(0, 5, n_samples)
    )
    
    # Ensure balanced distribution by using percentiles
    percentile_33 = np.percentile(risk_scores, 33.33)
    percentile_66 = np.percentile(risk_scores, 66.66)
    
    risk_levels = []
    for score in risk_scores:
        if score <= percentile_33:
            risk_levels.append('Stable')
        elif score <= percentile_66:
            risk_levels.append('Borderline')
        else:
            risk_levels.append('Unstable')
    
    data['risk_level'] = risk_levels
    data['profile_image'] = [f'profile_{i:04d}.jpg' for i in range(n_samples)]
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv('/workspace/sample_avalanche_data.csv', index=False)
    
    # Create sample images directory
    os.makedirs('/workspace/sample_images', exist_ok=True)
    
    # Create a few sample images for demonstration
    for i in range(min(10, n_samples)):
        # Create a synthetic snow profile image
        img = np.random.randint(0, 256, (200, 150, 3), dtype=np.uint8)
        
        # Add some structure to make it look like a snow profile
        for y in range(0, 200, 20):
            color = np.random.randint(100, 255)
            img[y:y+10, :] = color
        
        cv2.imwrite(f'/workspace/sample_images/profile_{i:04d}.jpg', img)
    
    print(f"Sample dataset created:")
    print(f"- CSV: /workspace/sample_avalanche_data.csv ({len(df)} samples)")
    print(f"- Images: /workspace/sample_images/ (10 sample images)")
    print(f"- Features: {len([col for col in df.columns if col not in ['risk_level', 'profile_image']])}")
    print(f"- Risk distribution: {df['risk_level'].value_counts().to_dict()}")


def main():
    """
    Main function to run the complete avalanche risk classification pipeline.
    """
    print("="*60)
    print("AVALANCHE RISK CLASSIFICATION PIPELINE")
    print("="*60)
    
    # Configuration
    csv_path = '/workspace/sample_avalanche_data.csv'
    image_dir = '/workspace/sample_images'
    
    # Create sample data if it doesn't exist
    if not os.path.exists(csv_path):
        create_sample_data()
    
    try:
        # Initialize classifier
        classifier = AvalancheRiskClassifier(csv_path, image_dir)
        
        # Step 1: Load and preprocess CSV data
        df = pd.read_csv(csv_path)
        structured_features, target = classifier.load_and_preprocess_csv()
        
        # Step 2: Extract OpenCV features from images
        opencv_features = classifier.extract_features_from_all_images(df)
        
        # Step 3: Combine features
        combined_features = classifier.combine_features(structured_features, opencv_features)
        
        # Step 4: Train model
        results = classifier.train_model(combined_features, target)
        
        # Step 5: Evaluate model
        classifier.evaluate_model(results)
        
        # Step 6: Save model
        classifier.save_model()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final model accuracy: {results['accuracy']:.4f}")
        print(f"Model saved as: /workspace/avalanche_model.pkl")
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        raise


if __name__ == '__main__':
    main()