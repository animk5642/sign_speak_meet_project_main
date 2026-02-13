#!/bin/bash
# Sign Language Recognition - Quick Setup Script
# Run this script to install dependencies and verify setup

echo "=================================="
echo "Sign Language Integration Setup"
echo "=================================="
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Virtual environment not activated!"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment detected: $VIRTUAL_ENV"
echo ""

# Install ML dependencies
echo "📦 Installing ML dependencies..."
pip install --quiet --upgrade pip
pip install --quiet tensorflow==2.13.0 numpy==1.24.3 opencv-python==4.8.1.78 \
    mediapipe==0.10.9 protobuf==3.20.3 pandas==2.0.3 \
    absl-py flatbuffers sounddevice

if [ $? -eq 0 ]; then
    echo "✅ ML dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo ""

# Verify model files
echo "🔍 Verifying model files..."
MODEL_FILE="sign-language-recognition/weights/model.tflite"
CSV_FILE="sign-language-recognition/asl-signs/train.csv"

if [ -f "$MODEL_FILE" ]; then
    SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "✅ Model file found: $MODEL_FILE ($SIZE)"
else
    echo "❌ Model file not found: $MODEL_FILE"
    echo "   Please ensure the model.tflite file exists"
    exit 1
fi

if [ -f "$CSV_FILE" ]; then
    LINES=$(wc -l < "$CSV_FILE")
    echo "✅ Training CSV found: $CSV_FILE ($LINES lines)"
else
    echo "❌ Training CSV not found: $CSV_FILE"
    exit 1
fi
echo ""

# Verify directory structure
echo "📁 Verifying directory structure..."
if [ -d "video_app/ml_service" ]; then
    echo "✅ ML service module exists"
else
    echo "❌ ML service module not found"
    exit 1
fi

if [ -f "video_app/ml_service/__init__.py" ]; then
    echo "✅ __init__.py exists"
else
    echo "❌ __init__.py missing - creating..."
    echo '"""ML Service Module for Sign Language Recognition"""' > video_app/ml_service/__init__.py
fi

if [ -f "video_app/ml_service/sign_language_detector.py" ]; then
    echo "✅ sign_language_detector.py exists"
else
    echo "❌ sign_language_detector.py missing"
    exit 1
fi

if [ -f "video_app/ml_service/config.py" ]; then
    echo "✅ config.py exists"
else
    echo "❌ config.py missing"
    exit 1
fi
echo ""

# Collect static files
echo "📦 Collecting static files..."
python manage.py collectstatic --noinput --clear > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Static files collected"
else
    echo "⚠️  Static files collection failed (this may be OK)"
fi
echo ""

# Run migrations (just in case)
echo "🗄️  Running migrations..."
python manage.py migrate --noinput > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Database migrations applied"
else
    echo "⚠️  Migrations may have issues"
fi
echo ""

# Test import
echo "🧪 Testing Python imports..."
python -c "
import sys
try:
    import tensorflow as tf
    import cv2
    import mediapipe as mp
    import numpy as np
    import pandas as pd
    print('✅ All ML libraries imported successfully')
    print(f'   TensorFlow: {tf.__version__}')
    print(f'   OpenCV: {cv2.__version__}')
    print(f'   MediaPipe: {mp.__version__}')
    print(f'   NumPy: {np.__version__}')
    print(f'   Pandas: {pd.__version__}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Python import test failed"
    exit 1
fi
echo ""

# Final summary
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Start the server: python manage.py runserver"
echo "2. Open a meeting room in your browser"
echo "3. Click the 🤲 (hands) button to enable sign language detection"
echo "4. Start making sign language gestures!"
echo ""
echo "📚 For more details, see: SIGN_LANGUAGE_INTEGRATION_GUIDE.md"
echo ""
