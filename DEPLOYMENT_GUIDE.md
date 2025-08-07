# 🏔️ Real-World Deployment Guide
## Avalanche Risk Assessment Pipeline

This guide shows you how to deploy the avalanche risk classification pipeline for real-world use with actual snowpack data and snow profile images.

## 🎯 Overview

The production pipeline integrates:
- **Real snowpack measurements** from field observations
- **Actual snow profile images** with enhanced OpenCV analysis
- **Weather data integration** from APIs
- **Database storage** for historical tracking
- **Real-time risk assessment** and reporting

## 📊 Data Collection Requirements

### 1. **Snowpack Measurements** (Field Observations)

#### **Essential Measurements:**
```python
# Snow characteristics
total_depth: float          # Total snow depth (cm)
new_snow_24h: float        # New snow in 24h (cm)
snow_density: float        # Snow density (g/cm³, typically 0.1-0.8)
temperature_surface: float # Surface temperature (°C)
temperature_10cm: float    # Temperature at 10cm depth (°C)

# Stability tests
compression_test_score: int    # CT score (taps to failure)
extended_column_test: str     # ECT result (ECTN, ECTP, etc.)
shear_strength: float         # Shear strength (kPa)

# Environmental
elevation: float              # Elevation (m)
slope_angle: float           # Slope angle (degrees)
aspect: str                  # Compass direction (N, NE, E, etc.)
```

#### **Equipment Needed:**
- **Snow Study Kit**: Shovel, snow saw, thermometer
- **Density Kit**: Snow tube, scale
- **Hardness Gauge**: For grain hardness measurement
- **Inclinometer**: For slope angle
- **Camera**: For snow profile images
- **GPS Device**: For location/elevation

### 2. **Snow Profile Images**

#### **Image Requirements:**
- **Resolution**: Minimum 1024x768 pixels
- **Format**: JPEG or PNG
- **Lighting**: Even, diffused lighting (avoid shadows)
- **Perspective**: Perpendicular to snow face
- **Scale Reference**: Include ruler or measurement tool

#### **Best Practices:**
```bash
# Image naming convention
location_YYYYMMDD_HHMM_profileN.jpg
# Example: whistler_20241215_1430_profile1.jpg
```

### 3. **Weather Data Integration**

#### **Automatic Collection:**
```python
# From weather APIs (OpenWeatherMap, Weather Underground, etc.)
air_temperature: float       # Air temperature (°C)
wind_speed: float           # Wind speed (km/h)
wind_direction: str         # Wind direction
precipitation_24h: float    # 24h precipitation (mm)
humidity: float             # Relative humidity (%)
```

#### **API Setup Example:**
```python
import requests

def get_weather_data(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric'
    }
    response = requests.get(url, params=params)
    return response.json()
```

## 🏗️ Production Setup

### 1. **Environment Setup**
```bash
# Clone the repository
git clone <your-repo>
cd avalanche-classification

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Additional production dependencies
pip install flask fastapi uvicorn SQLAlchemy
```

### 2. **Database Configuration**
```python
# For production, use PostgreSQL instead of SQLite
from sqlalchemy import create_engine
import os

# Database URL
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/avalanche_db')

# Setup
engine = create_engine(DATABASE_URL)
```

### 3. **Model Training with Real Data**

#### **Data Preparation:**
```python
from production_deployment_guide import RealWorldAvalanchePipeline

# Initialize pipeline
pipeline = RealWorldAvalanchePipeline()

# Load your real avalanche data
# Format: CSV with columns matching SnowpackMeasurement fields
real_data = pd.read_csv('real_avalanche_data.csv')

# Train model with real data
classifier = AvalancheRiskClassifier('real_data.csv', 'real_images/')
# ... training process
```

#### **Historical Data Sources:**
- **Avalanche Canada**: Historical avalanche bulletins
- **CAIC**: Colorado Avalanche Information Center data
- **EAWS**: European Avalanche Warning Services
- **Research Institutions**: University snow science programs
- **Ski Patrols**: Resort avalanche control records

## 🚀 Deployment Options

### 1. **Web API Deployment**

```python
from flask import Flask, request, jsonify
from production_deployment_guide import RealWorldAvalanchePipeline

app = Flask(__name__)
pipeline = RealWorldAvalanchePipeline('/path/to/trained_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_risk():
    data = request.json
    
    # Convert JSON to SnowpackMeasurement
    measurement = SnowpackMeasurement(**data)
    
    # Make prediction
    result = pipeline.predict_avalanche_risk(measurement)
    
    return jsonify(result)

@app.route('/report/<location_id>')
def get_risk_report(location_id):
    report = pipeline.generate_risk_report(location_id)
    return jsonify(report)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 2. **Docker Deployment**

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "api.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  avalanche-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/avalanche
    depends_on:
      - db
      
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: avalanche
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### 3. **Mobile Field App**

```python
# mobile_app.py - Simple field data collection
import streamlit as st
from datetime import datetime
import pandas as pd

st.title("🏔️ Avalanche Risk Assessment")

# Location input
col1, col2 = st.columns(2)
with col1:
    location_id = st.text_input("Location ID")
    elevation = st.number_input("Elevation (m)", min_value=0)
    slope_angle = st.number_input("Slope Angle (°)", min_value=0, max_value=90)

with col2:
    aspect = st.selectbox("Aspect", ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    timestamp = st.datetime_input("Date/Time", datetime.now())

# Snow measurements
st.subheader("Snow Measurements")
col1, col2 = st.columns(2)
with col1:
    total_depth = st.number_input("Total Depth (cm)", min_value=0.0)
    new_snow_24h = st.number_input("New Snow 24h (cm)", min_value=0.0)
    snow_density = st.number_input("Snow Density (g/cm³)", min_value=0.1, max_value=0.8)

with col2:
    temp_surface = st.number_input("Surface Temp (°C)", value=-5.0)
    temp_10cm = st.number_input("10cm Temp (°C)", value=-3.0)
    shear_strength = st.number_input("Shear Strength (kPa)", min_value=0.0)

# Image upload
uploaded_file = st.file_uploader("Snow Profile Image", type=['jpg', 'jpeg', 'png'])

if st.button("Assess Risk"):
    # Create measurement object and predict
    # ... implementation
    st.success("Risk assessment completed!")
```

## 📱 Integration Examples

### 1. **Weather Station Integration**
```python
# Automatic data collection from weather stations
class WeatherStationIntegration:
    def __init__(self, station_id, api_key):
        self.station_id = station_id
        self.api_key = api_key
    
    def collect_hourly_data(self):
        # Fetch data from weather station API
        # Process and standardize format
        # Store in database
        pass
    
    def trigger_assessment(self):
        # Check if conditions warrant risk assessment
        # Trigger pipeline if significant changes detected
        pass
```

### 2. **Satellite Data Integration**
```python
# Using satellite snow cover data
from sentinelsat import SentinelAPI

def get_satellite_snow_data(lat, lon, date):
    api = SentinelAPI('username', 'password')
    
    # Query for Sentinel-2 snow imagery
    products = api.query(
        area=f'POINT({lon} {lat})',
        date=date,
        producttype='S2MSI1C',
        cloudcoverpercentage=(0, 20)
    )
    
    return products
```

### 3. **Avalanche Bulletin Integration**
```python
# Parse existing avalanche bulletins for training data
import requests
from bs4 import BeautifulSoup

def scrape_avalanche_bulletins(region):
    """Extract historical avalanche data from bulletins"""
    # Implementation depends on data source format
    pass
```

## 📊 Data Quality & Validation

### 1. **Field Measurement Validation**
```python
def validate_measurement(measurement: SnowpackMeasurement) -> List[str]:
    """Validate field measurements for data quality"""
    warnings = []
    
    # Check reasonable ranges
    if measurement.snow_density and (measurement.snow_density < 0.1 or measurement.snow_density > 0.8):
        warnings.append("Snow density outside typical range (0.1-0.8 g/cm³)")
    
    if measurement.slope_angle and measurement.slope_angle > 60:
        warnings.append("Extreme slope angle - verify measurement")
    
    if measurement.temperature_surface > 5:
        warnings.append("Surface temperature above freezing - check conditions")
    
    return warnings
```

### 2. **Image Quality Assessment**
```python
def assess_image_quality(image_path: str) -> Dict:
    """Assess quality of snow profile image"""
    image = cv2.imread(image_path)
    
    quality_metrics = {
        'resolution_adequate': image.shape[0] >= 768 and image.shape[1] >= 1024,
        'brightness_ok': 50 <= np.mean(image) <= 200,
        'contrast_ok': np.std(image) >= 30,
        'focus_score': cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    }
    
    return quality_metrics
```

## ⚠️ Safety Considerations

### 1. **Model Limitations**
- **Not a replacement** for expert judgment
- **Use as supplementary tool** to human assessment
- **Always cross-reference** with local avalanche bulletins
- **Consider multiple data sources** before making decisions

### 2. **Deployment Checklist**
```python
DEPLOYMENT_CHECKLIST = {
    "Model Performance": [
        "✓ Trained on local/regional data",
        "✓ Validated against expert assessments", 
        "✓ Performance metrics documented",
        "✓ Regular model retraining scheduled"
    ],
    "Data Pipeline": [
        "✓ Automated data collection tested",
        "✓ Data validation rules implemented",
        "✓ Error handling and logging configured",
        "✓ Backup and recovery procedures"
    ],
    "User Interface": [
        "✓ Clear risk communication",
        "✓ Confidence intervals displayed",
        "✓ Contributing factors explained",
        "✓ Appropriate disclaimers included"
    ]
}
```

## 🔄 Continuous Improvement

### 1. **Model Retraining Pipeline**
```python
def scheduled_retraining():
    """Automated model retraining with new data"""
    # Collect new labeled data
    # Retrain model with expanded dataset
    # Validate performance on test set
    # Deploy if performance improves
    pass
```

### 2. **Performance Monitoring**
```python
def monitor_prediction_accuracy():
    """Track prediction accuracy over time"""
    # Compare predictions to actual avalanche occurrences
    # Generate performance reports
    # Alert if accuracy drops below threshold
    pass
```

## 📞 Support & Resources

### **Professional Organizations:**
- **American Avalanche Association (AAA)**
- **Canadian Avalanche Association (CAA)**
- **European Avalanche Warning Services (EAWS)**

### **Training & Certification:**
- **Avalanche Level 1/2/3 Courses**
- **Snow Science Workshops**
- **Professional Observer Programs**

### **Data Sources:**
- **SNOTEL Network** (US)
- **Avalanche Canada Database**
- **Global Summary of the Month (GSOM)**
- **Research Data Archives**

---

## 🚀 Getting Started

1. **Start with the example**: Run `python production_deployment_guide.py`
2. **Collect sample data**: Begin with a small dataset from your region
3. **Train regional model**: Use local avalanche history and snow conditions
4. **Deploy incrementally**: Start with internal testing, then expand
5. **Monitor and improve**: Continuously validate and refine the model

**Remember**: This system augments, but never replaces, professional avalanche assessment and local expertise!