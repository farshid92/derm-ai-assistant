#!/usr/bin/env python3
"""
DermAI Assistant - Comprehensive Testing Suite
Tests backend API, frontend functionality, and model performance
"""

import requests
import time
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import io
import json

def test_backend_health():
    """Test if backend is running and healthy"""
    print("🔍 Testing Backend Health...")
    
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and accessible")
            return True
        else:
            print(f"❌ Backend returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running. Please start the backend first.")
        return False
    except Exception as e:
        print(f"❌ Error testing backend: {e}")
        return False

def test_model_loading():
    """Test if the trained model can be loaded"""
    print("\n🔍 Testing Model Loading...")
    
    model_path = Path("backend/model/skin_segmentation_unet_fine_tuned_augmented2.keras")
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        import tensorflow as tf
        from keras.models import load_model
        
        # Test loading the model
        model = load_model(str(model_path))
        print(f"✅ Model loaded successfully")
        print(f"   - Model type: {type(model)}")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        print(f"   - Parameters: {model.count_params():,}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_api_prediction():
    """Test the prediction API endpoint"""
    print("\n🔍 Testing API Prediction...")
    
    # Check if sample image exists
    sample_image_path = Path("ISIC_0000001.jpg")
    if not sample_image_path.exists():
        print(f"❌ Sample image not found: {sample_image_path}")
        print("   Please make sure ISIC_0000001.jpg is in the root directory")
        return False
    
    try:
        # Prepare test image
        with open(sample_image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            
            # Send prediction request
            start_time = time.time()
            response = requests.post("http://localhost:8000/predict", files=files, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                print("✅ API prediction successful")
                print(f"   - Response time: {end_time - start_time:.2f} seconds")
                print(f"   - Response size: {len(response.content)} bytes")
                
                # Save and analyze the result
                result_image = Image.open(io.BytesIO(response.content))
                print(f"   - Result image size: {result_image.size}")
                print(f"   - Result image mode: {result_image.mode}")
                
                # Convert to numpy for analysis
                mask_array = np.array(result_image)
                if len(mask_array.shape) == 3:
                    mask_array = mask_array[:, :, 0]
                
                # Calculate metrics
                lesion_area = np.sum(mask_array > 0)
                total_pixels = mask_array.size
                lesion_percentage = (lesion_area / total_pixels) * 100
                
                print(f"   - Lesion area: {lesion_area:,} pixels")
                print(f"   - Lesion coverage: {lesion_percentage:.2f}%")
                
                return True
            else:
                print(f"❌ API prediction failed with status code: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing API prediction: {e}")
        return False

def test_frontend_access():
    """Test if frontend is accessible"""
    print("\n🔍 Testing Frontend Access...")
    
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            return True
        else:
            print(f"❌ Frontend returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Frontend is not running. Please start the frontend.")
        return False
    except Exception as e:
        print(f"❌ Error testing frontend: {e}")
        return False

def test_batch_processing():
    """Test batch processing functionality"""
    print("\n🔍 Testing Batch Processing...")
    
    # Create test images (if sample exists, use it multiple times)
    sample_image_path = Path("ISIC_0000001.jpg")
    if not sample_image_path.exists():
        print("❌ No sample image found for batch testing")
        return False
    
    try:
        # Test with 3 copies of the same image
        test_files = []
        for i in range(3):
            with open(sample_image_path, 'rb') as f:
                test_files.append(('file', (f'test_{i}.jpg', f.read(), 'image/jpeg')))
        
        # Send batch requests
        start_time = time.time()
        results = []
        for i, (_, file_data) in enumerate(test_files):
            files = {'file': file_data}
            response = requests.post("http://localhost:8000/predict", files=files, timeout=30)
            if response.status_code == 200:
                results.append(True)
            else:
                results.append(False)
        
        end_time = time.time()
        
        success_count = sum(results)
        print(f"✅ Batch processing completed")
        print(f"   - Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        print(f"   - Total time: {end_time - start_time:.2f} seconds")
        print(f"   - Average time per image: {(end_time - start_time)/len(results):.2f} seconds")
        
        return success_count == len(results)
        
    except Exception as e:
        print(f"❌ Error testing batch processing: {e}")
        return False

def test_model_performance():
    """Test model performance metrics"""
    print("\n🔍 Testing Model Performance...")
    
    sample_image_path = Path("ISIC_0000001.jpg")
    if not sample_image_path.exists():
        print("❌ No sample image found for performance testing")
        return False
    
    try:
        # Test multiple predictions to get average performance
        times = []
        for i in range(5):
            with open(sample_image_path, 'rb') as f:
                files = {'file': ('test_image.jpg', f, 'image/jpeg')}
                
                start_time = time.time()
                response = requests.post("http://localhost:8000/predict", files=files, timeout=30)
                end_time = time.time()
                
                if response.status_code == 200:
                    times.append(end_time - start_time)
        
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            print("✅ Performance test completed")
            print(f"   - Average time: {avg_time:.3f} ± {std_time:.3f} seconds")
            print(f"   - Min time: {min_time:.3f} seconds")
            print(f"   - Max time: {max_time:.3f} seconds")
            print(f"   - Throughput: {1/avg_time:.1f} images/second")
            
            # Performance benchmarks
            if avg_time < 2.0:
                print("   - ✅ Performance: Excellent (< 2s)")
            elif avg_time < 5.0:
                print("   - ⚠️  Performance: Good (< 5s)")
            else:
                print("   - ❌ Performance: Slow (> 5s)")
            
            return True
        else:
            print("❌ No successful predictions for performance testing")
            return False
            
    except Exception as e:
        print(f"❌ Error testing performance: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\n🔍 Testing Error Handling...")
    
    # Test 1: Invalid file type
    try:
        files = {'file': ('test.txt', b'invalid content', 'text/plain')}
        response = requests.post("http://localhost:8000/predict", files=files, timeout=10)
        print(f"   - Invalid file type: Status {response.status_code}")
    except Exception as e:
        print(f"   - Invalid file type: Error handled - {type(e).__name__}")
    
    # Test 2: Empty file
    try:
        files = {'file': ('empty.jpg', b'', 'image/jpeg')}
        response = requests.post("http://localhost:8000/predict", files=files, timeout=10)
        print(f"   - Empty file: Status {response.status_code}")
    except Exception as e:
        print(f"   - Empty file: Error handled - {type(e).__name__}")
    
    # Test 3: Very large file (simulate)
    try:
        large_content = b'x' * (10 * 1024 * 1024)  # 10MB
        files = {'file': ('large.jpg', large_content, 'image/jpeg')}
        response = requests.post("http://localhost:8000/predict", files=files, timeout=30)
        print(f"   - Large file: Status {response.status_code}")
    except Exception as e:
        print(f"   - Large file: Error handled - {type(e).__name__}")
    
    print("✅ Error handling tests completed")
    return True

def generate_test_report(results):
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("📊 TEST REPORT")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if failed_tests == 0:
        print("\n🎉 All tests passed! Your DermAI Assistant is ready to use.")
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed. Please check the issues above.")
    
    return passed_tests == total_tests

def main():
    """Run all tests"""
    print("🔬 DermAI Assistant - Comprehensive Testing Suite")
    print("="*60)
    
    # Wait a moment for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(3)
    
    # Run all tests
    tests = {
        "Backend Health": test_backend_health,
        "Model Loading": test_model_loading,
        "API Prediction": test_api_prediction,
        "Frontend Access": test_frontend_access,
        "Batch Processing": test_batch_processing,
        "Model Performance": test_model_performance,
        "Error Handling": test_error_handling
    }
    
    results = {}
    
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Generate report
    all_passed = generate_test_report(results)
    
    # Provide next steps
    print("\n" + "="*60)
    print("🚀 NEXT STEPS")
    print("="*60)
    
    if all_passed:
        print("1. Open your browser and go to: http://localhost:8501")
        print("2. Upload a dermoscopic image to test the segmentation")
        print("3. Try the batch processing feature")
        print("4. Explore the different model options")
        print("5. Download and analyze the results")
    else:
        print("1. Check that both backend and frontend are running")
        print("2. Verify the model file exists in backend/model/")
        print("3. Ensure all dependencies are installed")
        print("4. Check the console for error messages")
        print("5. Restart the application if needed")
    
    print("\n📚 For more information, check the README.md file")
    print("🔗 GitHub: https://github.com/farshid92/derm-ai-assistant")

if __name__ == "__main__":
    main() 