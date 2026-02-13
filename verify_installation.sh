#!/bin/bash
# Installation and Verification Checklist
# Run this to verify everything is set up correctly

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Sign Language Recognition - Installation Checklist         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

PASS="✅"
FAIL="❌"
WARN="⚠️"

# Track success
ALL_GOOD=true

echo "1. CHECKING VIRTUAL ENVIRONMENT..."
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "   $PASS Virtual environment activated: $VIRTUAL_ENV"
else
    echo "   $FAIL Virtual environment not activated"
    echo "   Run: source venv/bin/activate"
    ALL_GOOD=false
fi
echo ""

echo "2. CHECKING PYTHON DEPENDENCIES..."
python -c "import tensorflow" 2>/dev/null && echo "   $PASS TensorFlow installed" || { echo "   $FAIL TensorFlow not installed"; ALL_GOOD=false; }
python -c "import mediapipe" 2>/dev/null && echo "   $PASS MediaPipe installed" || { echo "   $FAIL MediaPipe not installed"; ALL_GOOD=false; }
python -c "import cv2" 2>/dev/null && echo "   $PASS OpenCV installed" || { echo "   $FAIL OpenCV not installed"; ALL_GOOD=false; }
python -c "import numpy" 2>/dev/null && echo "   $PASS NumPy installed" || { echo "   $FAIL NumPy not installed"; ALL_GOOD=false; }
python -c "import pandas" 2>/dev/null && echo "   $PASS Pandas installed" || { echo "   $FAIL Pandas not installed"; ALL_GOOD=false; }
echo ""

echo "3. CHECKING MODEL FILES..."
if [ -f "sign-language-recognition/weights/model.tflite" ]; then
    SIZE=$(du -h "sign-language-recognition/weights/model.tflite" | cut -f1)
    echo "   $PASS model.tflite found ($SIZE)"
else
    echo "   $FAIL model.tflite not found"
    ALL_GOOD=false
fi

if [ -f "sign-language-recognition/asl-signs/train.csv" ]; then
    LINES=$(wc -l < "sign-language-recognition/asl-signs/train.csv")
    echo "   $PASS train.csv found ($LINES lines)"
else
    echo "   $FAIL train.csv not found"
    ALL_GOOD=false
fi
echo ""

echo "4. CHECKING PROJECT STRUCTURE..."
[ -d "video_app/ml_service" ] && echo "   $PASS ml_service/ directory exists" || { echo "   $FAIL ml_service/ directory missing"; ALL_GOOD=false; }
[ -f "video_app/ml_service/__init__.py" ] && echo "   $PASS __init__.py exists" || { echo "   $FAIL __init__.py missing"; ALL_GOOD=false; }
[ -f "video_app/ml_service/config.py" ] && echo "   $PASS config.py exists" || { echo "   $FAIL config.py missing"; ALL_GOOD=false; }
[ -f "video_app/ml_service/sign_language_detector.py" ] && echo "   $PASS sign_language_detector.py exists" || { echo "   $FAIL sign_language_detector.py missing"; ALL_GOOD=false; }
echo ""

echo "5. CHECKING MODIFIED FILES..."
grep -q "SignLanguageConsumer" video_app/consumers.py && echo "   $PASS consumers.py updated" || { echo "   $FAIL consumers.py not updated"; ALL_GOOD=false; }
grep -q "SignLanguageManager" static/js/main.js && echo "   $PASS main.js updated" || { echo "   $FAIL main.js not updated"; ALL_GOOD=false; }
grep -q "signLanguageBtn" templates/video_app/meeting_room.html && echo "   $PASS meeting_room.html updated" || { echo "   $FAIL meeting_room.html not updated"; ALL_GOOD=false; }
grep -q "tensorflow" requirements.txt && echo "   $PASS requirements.txt updated" || { echo "   $FAIL requirements.txt not updated"; ALL_GOOD=false; }
echo ""

echo "6. CHECKING WEBSOCKET ROUTING..."
grep -q "sign-language" video_app/routing.py && echo "   $PASS WebSocket route configured" || { echo "   $FAIL WebSocket route missing"; ALL_GOOD=false; }
echo ""

echo "7. CHECKING DOCUMENTATION..."
[ -f "SIGN_LANGUAGE_INTEGRATION_GUIDE.md" ] && echo "   $PASS Integration guide exists" || echo "   $WARN Integration guide missing"
[ -f "SIGN_LANGUAGE_QUICK_REFERENCE.md" ] && echo "   $PASS Quick reference exists" || echo "   $WARN Quick reference missing"
[ -f "IMPLEMENTATION_SUMMARY.md" ] && echo "   $PASS Implementation summary exists" || echo "   $WARN Implementation summary missing"
[ -f "README_COMPLETE.md" ] && echo "   $PASS Complete README exists" || echo "   $WARN Complete README missing"
echo ""

echo "8. TESTING IMPORTS..."
python -c "from video_app.ml_service.sign_language_detector import SignLanguageDetector" 2>/dev/null && echo "   $PASS ML module imports successfully" || { echo "   $FAIL ML module import failed"; ALL_GOOD=false; }
python -c "from video_app.ml_service.config import MODEL_PATH" 2>/dev/null && echo "   $PASS Config module imports successfully" || { echo "   $FAIL Config module import failed"; ALL_GOOD=false; }
echo ""

echo "9. CHECKING DJANGO SETUP..."
python manage.py check --deploy 2>/dev/null >/dev/null && echo "   $PASS Django checks passed" || echo "   $WARN Django checks have warnings (may be OK)"
echo ""

echo "═══════════════════════════════════════════════════════════════"
if [ "$ALL_GOOD" = true ]; then
    echo "║  ✅ ALL CHECKS PASSED - READY TO GO! ✅                     ║"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo "1. python manage.py runserver"
    echo "2. Open a meeting in your browser"
    echo "3. Click the 🤲 button to enable sign language detection"
    echo "4. Start signing!"
    echo ""
else
    echo "║  ❌ SOME CHECKS FAILED - PLEASE FIX ISSUES ABOVE ❌         ║"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Common fixes:"
    echo "1. Install dependencies: ./setup_sign_language.sh"
    echo "2. Verify model files exist"
    echo "3. Check file modifications"
    echo ""
fi

echo "For detailed help, see: SIGN_LANGUAGE_INTEGRATION_GUIDE.md"
echo ""
