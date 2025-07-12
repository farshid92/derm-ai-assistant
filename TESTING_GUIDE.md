# 🔬 DermAI Assistant - Testing Guide

This guide will help you test your DermAI Assistant application thoroughly to ensure everything is working correctly.

## 🚀 Quick Start Testing

### 1. Automated Testing
Run the comprehensive test suite:
```bash
python test_app.py
```

This will test:
- ✅ Backend health and accessibility
- ✅ Model loading and configuration
- ✅ API prediction functionality
- ✅ Frontend accessibility
- ✅ Batch processing capabilities
- ✅ Model performance metrics
- ✅ Error handling

### 2. Manual Testing Steps

#### Step 1: Start the Application
```bash
python run_app.py
```

Wait for both services to start:
- Backend: http://localhost:8000
- Frontend: http://localhost:8501

#### Step 2: Test Backend API
1. **Open API Documentation**: Go to http://localhost:8000/docs
2. **Test Prediction Endpoint**:
   - Click on POST /predict
   - Click "Try it out"
   - Upload a test image (ISIC_0000001.jpg)
   - Click "Execute"
   - Verify you get a segmentation mask back

#### Step 3: Test Frontend Interface
1. **Open Frontend**: Go to http://localhost:8501
2. **Test Single Image Processing**:
   - Upload ISIC_0000001.jpg
   - Click "Analyze Image"
   - Verify segmentation results appear
   - Check lesion area calculations
   - Test download functionality

3. **Test Batch Processing**:
   - Navigate to Batch Processing tab
   - Upload multiple images
   - Process batch and verify results
   - Download CSV report

#### Step 4: Test Different Scenarios
1. **Model Selection**: Try different model options in sidebar
2. **Parameter Adjustment**: Test confidence threshold slider
3. **Error Handling**: Try uploading invalid files
4. **Performance**: Test with different image sizes

## 📊 Expected Results

### Model Performance
- **Processing Time**: < 2 seconds per image
- **Accuracy**: ~88.5% (Dice score: 0.885)
- **Memory Usage**: Efficient for batch processing

### Image Requirements
- **Format**: JPG, JPEG, PNG
- **Size**: Any size (will be resized to 128x128)
- **Color**: RGB (will be converted if needed)

### Output Quality
- **Segmentation Mask**: Binary image (black/white)
- **Overlay**: Original image with red lesion highlighting
- **Metrics**: Lesion area and coverage percentage

## 🧪 Advanced Testing

### 1. Performance Testing
```bash
# Test multiple images for performance
python -c "
import time
import requests
from pathlib import Path

times = []
for i in range(10):
    with open('ISIC_0000001.jpg', 'rb') as f:
        start = time.time()
        response = requests.post('http://localhost:8000/predict', 
                               files={'file': f})
        times.append(time.time() - start)

print(f'Average time: {sum(times)/len(times):.2f}s')
print(f'Min time: {min(times):.2f}s')
print(f'Max time: {max(times):.2f}s')
"
```

### 2. Model Validation
```bash
# Test model loading and basic inference
python -c "
from keras.models import load_model
import numpy as np
from PIL import Image

# Load model
model = load_model('backend/model/skin_segmentation_unet_fine_tuned_augmented2.keras')

# Test with sample image
img = Image.open('ISIC_0000001.jpg').resize((128, 128))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
print(f'Prediction shape: {pred.shape}')
print(f'Prediction range: {pred.min():.3f} to {pred.max():.3f}')
print(f'Prediction mean: {pred.mean():.3f}')
"
```

### 3. API Testing with curl
```bash
# Test API with curl
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@ISIC_0000001.jpg" \
     --output test_result.png

# Check if result was generated
ls -la test_result.png
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Backend Not Starting
**Symptoms**: Connection refused on port 8000
**Solutions**:
- Check if port 8000 is already in use
- Verify TensorFlow is installed correctly
- Check model file exists in backend/model/

#### 2. Frontend Not Loading
**Symptoms**: Connection refused on port 8501
**Solutions**:
- Check if Streamlit is installed
- Verify all frontend dependencies are installed
- Check for Python version compatibility

#### 3. Model Loading Errors
**Symptoms**: Keras model loading fails
**Solutions**:
- Verify model file path: `backend/model/skin_segmentation_unet_fine_tuned_augmented2.keras`
- Check TensorFlow version compatibility
- Ensure model was saved correctly during training

#### 4. Slow Performance
**Symptoms**: Processing takes >5 seconds
**Solutions**:
- Check GPU availability
- Monitor system resources
- Consider model optimization

#### 5. Memory Issues
**Symptoms**: Out of memory errors
**Solutions**:
- Reduce batch size
- Close other applications
- Use smaller input images

### Debug Commands

```bash
# Check if services are running
netstat -an | grep :8000  # Backend
netstat -an | grep :8501  # Frontend

# Check model file
ls -la backend/model/

# Check dependencies
pip list | grep -E "(tensorflow|streamlit|fastapi)"

# Check logs
# Look for error messages in the terminal where you ran run_app.py
```

## 📈 Performance Benchmarks

### Expected Performance Metrics
- **Single Image**: 1-2 seconds
- **Batch Processing**: 2-3 seconds per image
- **Memory Usage**: <2GB RAM
- **CPU Usage**: <50% on modern systems
- **GPU Usage**: If available, should utilize GPU

### Performance Testing Script
```python
import time
import requests
import numpy as np
from pathlib import Path

def benchmark_performance():
    times = []
    for i in range(20):
        with open('ISIC_0000001.jpg', 'rb') as f:
            start = time.time()
            response = requests.post('http://localhost:8000/predict', 
                                   files={'file': f})
            times.append(time.time() - start)
    
    print(f"Average: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
    print(f"Min: {np.min(times):.3f}s")
    print(f"Max: {np.max(times):.3f}s")
    print(f"Throughput: {1/np.mean(times):.1f} images/second")

benchmark_performance()
```

## ✅ Success Criteria

Your DermAI Assistant is working correctly if:

1. **✅ Backend starts without errors**
2. **✅ Frontend loads in browser**
3. **✅ Model loads successfully**
4. **✅ Single image processing works**
5. **✅ Batch processing works**
6. **✅ Results are visually reasonable**
7. **✅ Download functionality works**
8. **✅ Performance is acceptable (<2s per image)**

## 🎯 Next Steps After Testing

Once testing is complete:

1. **Document Results**: Note any issues or performance metrics
2. **Optimize if Needed**: Address any performance bottlenecks
3. **Deploy**: Consider deploying to cloud platform
4. **Share**: Update GitHub repository with results
5. **Improve**: Plan next features or optimizations

## 📞 Getting Help

If you encounter issues:

1. **Check the logs** in the terminal where you ran `run_app.py`
2. **Verify dependencies** are installed correctly
3. **Check file paths** and permissions
4. **Review error messages** carefully
5. **Test with different images** to isolate issues

For more help, check the main README.md file or create an issue on GitHub.

---

**Happy Testing! 🎉** 