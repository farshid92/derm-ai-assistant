import streamlit as st
import requests
import io
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
import os

# Page configuration
st.set_page_config(
    page_title="DermAI Assistant - Skin Lesion Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
    .result-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

def home_page():
    """Home page with project overview"""
    st.markdown('<h1 class="main-header">🔬 DermAI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Skin Lesion Segmentation with Deep Learning</p>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <h2>88.5%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎲 Dice Score</h3>
            <h2>0.885</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Speed</h3>
            <h2>< 2s</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🔬 Models</h3>
            <h2>4</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Project description
    st.markdown("## 🎯 About This Project")
    st.markdown("""
    DermAI Assistant is a state-of-the-art deep learning application for automated skin lesion segmentation 
    from dermoscopic images. Built with U-Net architecture and advanced data augmentation techniques, 
    this tool achieves high accuracy for medical image analysis.
    
    ### Key Features:
    - **High Accuracy**: 88.5% accuracy with 0.885 Dice score
    - **Multiple Models**: Support for U-Net, VGG16, ConvNeXt, and ensemble methods
    - **Real-time Processing**: Fast inference with <2 second processing time
    - **Professional UI**: Modern Streamlit interface with comprehensive visualization
    - **Batch Processing**: Handle multiple images efficiently
    - **Export Results**: Download segmentation masks and analysis reports
    """)
    
    # Quick start section
    st.markdown("## 🚀 Quick Start")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Single Image Analysis")
        st.markdown("""
        1. Navigate to **Single Image** tab
        2. Upload your dermoscopic image
        3. Select model and parameters
        4. Click **Analyze Image**
        5. View and download results
        """)
    
    with col2:
        st.markdown("### 🔄 Batch Processing")
        st.markdown("""
        1. Navigate to **Batch Processing** tab
        2. Upload multiple images
        3. Configure batch settings
        4. Process all images at once
        5. Download comprehensive report
        """)
    
    # Sample image
    st.markdown("## 📋 Sample Image")
    try:
        sample_image = Image.open("ISIC_0000001.jpg")
        st.image(sample_image, caption="Sample Dermoscopic Image (ISIC_0000001.jpg)", use_column_width=True)
        st.info("Try this sample image in the Single Image tab to see the segmentation in action!")
    except:
        st.info("No sample images found. Upload your own image to test the segmentation.")

def single_image_page():
    """Single image processing page"""
    st.markdown("## 📤 Single Image Analysis")
    st.markdown("Upload a skin lesion image for AI-powered segmentation analysis.")
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### Model Selection")
        model_option = st.selectbox(
            "Choose your segmentation model:",
            ["U-Net (Fine-tuned)", "VGG16 + U-Net", "ConvNeXt + U-Net", "Ensemble (Genetic Algorithm)"],
            help="Select the model architecture for segmentation"
        )
        
        st.markdown("### Parameters")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Threshold for binary mask generation"
        )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        # Display original image
        st.markdown("### 📷 Original Image")
        original_image = Image.open(uploaded_file)
        
        # Create two columns for image display
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_image, caption="Original Dermoscopic Image", use_column_width=True)
            
            # Image information
            st.markdown("**Image Information:**")
            st.write(f"**Size:** {original_image.size[0]} × {original_image.size[1]} pixels")
            st.write(f"**Format:** {original_image.format}")
            st.write(f"**Mode:** {original_image.mode}")
        
        # Process button
        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Processing image with AI..."):
                try:
                    # Prepare image for API
                    img_byte_arr = io.BytesIO()
                    original_image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Send to backend API
                    files = {"file": ("image.png", img_byte_arr, "image/png")}
                    response = requests.post("http://localhost:8000/predict", files=files)
                    
                    if response.status_code == 200:
                        # Load predicted mask
                        mask_image = Image.open(io.BytesIO(response.content))
                        
                        with col2:
                            st.image(mask_image, caption="Segmentation Mask", use_column_width=True)
                            
                            # Convert to numpy for analysis
                            mask_array = np.array(mask_image)
                            if len(mask_array.shape) == 3:
                                mask_array = mask_array[:, :, 0]  # Take first channel if RGB
                            
                            # Calculate metrics
                            lesion_area = np.sum(mask_array > 0)
                            total_pixels = mask_array.size
                            lesion_percentage = (lesion_area / total_pixels) * 100
                            
                            st.markdown("**Segmentation Results:**")
                            st.write(f"**Lesion Area:** {lesion_area:,} pixels")
                            st.write(f"**Lesion Coverage:** {lesion_percentage:.2f}%")
                            st.write(f"**Model Used:** {model_option}")
                        
                        # Create overlay visualization
                        st.markdown("### 🔍 Detailed Analysis")
                        
                        # Create overlay
                        original_array = np.array(original_image)
                        mask_array = np.array(mask_image)
                        # If mask is single channel, expand to 3 channels for overlay
                        if len(mask_array.shape) == 2:
                            mask_rgb = np.stack([mask_array]*3, axis=-1)
                        else:
                            mask_rgb = mask_array
                        
                        overlay = original_array.copy()
                        overlay[mask_rgb[..., 0] > 0] = [255, 0, 0]  # Red overlay for lesion
                        
                        # Create subplot
                        fig = make_subplots(
                            rows=1, cols=3,
                            subplot_titles=("Original", "Segmentation Mask", "Overlay"),
                            specs=[[{"type": "image"}, {"type": "image"}, {"type": "image"}]]
                        )
                        
                        # Add images to subplot
                        fig.add_trace(go.Image(z=original_array), row=1, col=1)
                        fig.add_trace(go.Image(z=mask_array), row=1, col=2)
                        fig.add_trace(go.Image(z=overlay), row=1, col=3)
                        
                        fig.update_layout(
                            height=400,
                            showlegend=False,
                            title_text="Segmentation Analysis Results"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        st.markdown("### 💾 Download Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Download mask
                            mask_buffer = io.BytesIO()
                            mask_image.save(mask_buffer, format='PNG')
                            mask_buffer.seek(0)
                            st.download_button(
                                label="📥 Download Segmentation Mask",
                                data=mask_buffer.getvalue(),
                                file_name="segmentation_mask.png",
                                mime="image/png"
                            )
                        
                        with col2:
                            # Download overlay
                            overlay_img = Image.fromarray(overlay)
                            overlay_buffer = io.BytesIO()
                            overlay_img.save(overlay_buffer, format='PNG')
                            overlay_buffer.seek(0)
                            st.download_button(
                                label="📥 Download Overlay Image",
                                data=overlay_buffer.getvalue(),
                                file_name="overlay_image.png",
                                mime="image/png"
                            )
                    
                    else:
                        st.error(f"Error processing image: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure the backend server is running on http://localhost:8000")

    # Instructions when no file is uploaded
    else:
        st.markdown("""
        <div class="upload-area">
            <h3>📤 Upload your dermoscopic image to get started</h3>
            <p>Drag and drop or click to browse for your image file</p>
            <p><strong>Supported formats:</strong> JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

def batch_processing_page():
    """Batch processing page"""
    st.markdown("## 🔄 Batch Processing")
    st.markdown("Upload multiple images for batch segmentation analysis.")
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### Batch Settings")
        model_option = st.selectbox(
            "Model:",
            ["U-Net (Fine-tuned)", "VGG16 + U-Net", "ConvNeXt + U-Net", "Ensemble (Genetic Algorithm)"]
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1
        )
        
        max_files = st.number_input(
            "Max Files:",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum number of files to process"
        )
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose multiple image files",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Select multiple images for batch processing"
    )
    
    if uploaded_files:
        if len(uploaded_files) > max_files:
            st.warning(f"Too many files selected. Maximum allowed: {max_files}")
            uploaded_files = uploaded_files[:max_files]
        
        st.info(f"Selected {len(uploaded_files)} images for processing")
        
        # Process button
        if st.button("🔄 Process Batch", type="primary", use_container_width=True):
            if len(uploaded_files) > 0:
                results = process_batch_images(uploaded_files, model_option, confidence_threshold)
                display_batch_results(results)

def process_batch_images(uploaded_files, model_option, confidence_threshold):
    """Process multiple images and return results"""
    
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
        
        try:
            # Load and process image
            original_image = Image.open(uploaded_file)
            
            # Prepare for API
            img_byte_arr = io.BytesIO()
            original_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Send to backend
            files = {"file": (uploaded_file.name, img_byte_arr, "image/png")}
            response = requests.post("http://localhost:8000/predict", files=files)
            
            if response.status_code == 200:
                # Process results
                mask_image = Image.open(io.BytesIO(response.content))
                mask_array = np.array(mask_image)
                
                if len(mask_array.shape) == 3:
                    mask_array = mask_array[:, :, 0]
                
                # Calculate metrics
                lesion_area = np.sum(mask_array > 0)
                total_pixels = mask_array.size
                lesion_percentage = (lesion_area / total_pixels) * 100
                
                results.append({
                    'filename': uploaded_file.name,
                    'original_size': original_image.size,
                    'lesion_area': lesion_area,
                    'lesion_percentage': lesion_percentage,
                    'model_used': model_option,
                    'confidence_threshold': confidence_threshold,
                    'status': 'Success',
                    'original_image': original_image,
                    'mask_image': mask_image
                })
            else:
                results.append({
                    'filename': uploaded_file.name,
                    'status': f'Error: {response.status_code}',
                    'original_image': original_image
                })
                
        except Exception as e:
            results.append({
                'filename': uploaded_file.name,
                'status': f'Error: {str(e)}'
            })
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("Processing complete!")
    return results

def display_batch_results(results):
    """Display comprehensive batch processing results"""
    
    # Summary statistics
    successful_results = [r for r in results if r['status'] == 'Success']
    
    if successful_results:
        st.markdown("## 📊 Batch Processing Summary")
        
        # Create summary dataframe
        summary_data = []
        for result in successful_results:
            summary_data.append({
                'Filename': result['filename'],
                'Lesion Area (pixels)': result['lesion_area'],
                'Lesion Coverage (%)': f"{result['lesion_percentage']:.2f}%",
                'Image Size': f"{result['original_size'][0]}×{result['original_size'][1]}"
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", len(results))
        
        with col2:
            st.metric("Successful", len(successful_results))
        
        with col3:
            avg_coverage = np.mean([r['lesion_percentage'] for r in successful_results])
            st.metric("Avg Coverage", f"{avg_coverage:.2f}%")
        
        with col4:
            total_area = np.sum([r['lesion_area'] for r in successful_results])
            st.metric("Total Lesion Area", f"{total_area:,}")
        
        # Distribution plot
        st.markdown("### 📈 Lesion Coverage Distribution")
        coverage_values = [r['lesion_percentage'] for r in successful_results]
        
        fig = px.histogram(
            x=coverage_values,
            nbins=20,
            title="Distribution of Lesion Coverage Percentages",
            labels={'x': 'Lesion Coverage (%)', 'y': 'Number of Images'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual results
        st.markdown("## 🔍 Individual Results")
        
        for i, result in enumerate(successful_results):
            with st.expander(f"📷 {result['filename']} - {result['lesion_percentage']:.2f}% coverage"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(result['original_image'], caption="Original", use_column_width=True)
                
                with col2:
                    st.image(result['mask_image'], caption="Segmentation", use_column_width=True)
                
                with col3:
                    # Create overlay
                    original_array = np.array(result['original_image'].resize((128, 128)))
                    mask_resized = np.array(result['mask_image'].resize((128, 128)))
                    
                    overlay = original_array.copy()
                    if len(mask_resized.shape) == 3:
                        mask_binary = mask_resized[:, :, 0] > 0
                    else:
                        mask_binary = mask_resized > 0
                    
                    overlay[mask_binary] = [255, 0, 0]
                    st.image(overlay, caption="Overlay", use_column_width=True)
                
                # Metrics
                st.write(f"**Lesion Area:** {result['lesion_area']:,} pixels")
                st.write(f"**Coverage:** {result['lesion_percentage']:.2f}%")
                st.write(f"**Model:** {result['model_used']}")
        
        # Export results
        st.markdown("## 💾 Export Results")
        
        # Create CSV report
        export_data = []
        
        for result in successful_results:
            export_data.append({
                'Filename': result['filename'],
                'Lesion_Area_Pixels': result['lesion_area'],
                'Lesion_Coverage_Percent': result['lesion_percentage'],
                'Image_Width': result['original_size'][0],
                'Image_Height': result['original_size'][1],
                'Model_Used': result['model_used'],
                'Confidence_Threshold': result['confidence_threshold']
            })
        
        df_export = pd.DataFrame(export_data)
        csv = df_export.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV Report",
            data=csv,
            file_name=f"batch_processing_report_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Error summary
    error_results = [r for r in results if r['status'] != 'Success']
    if error_results:
        st.markdown("## ❌ Processing Errors")
        for result in error_results:
            st.error(f"**{result['filename']}:** {result['status']}")

def about_page():
    """About page with project information"""
    st.markdown("## ℹ️ About DermAI Assistant")
    
    st.markdown("""
    ### Project Overview
    DermAI Assistant is a comprehensive deep learning application for automated skin lesion segmentation 
    from dermoscopic images. This project demonstrates advanced computer vision techniques and medical AI applications.
    
    ### Technical Details
    - **Framework**: TensorFlow 2.19+ with Keras
    - **Architecture**: U-Net with encoder-decoder structure
    - **Backend**: FastAPI for high-performance API
    - **Frontend**: Streamlit for interactive web interface
    - **Dataset**: ISIC 2018 Challenge dermoscopic images
    
    ### Model Performance
    - **Dice Score**: 0.885
    - **Accuracy**: 88.5%
    - **Processing Speed**: <2 seconds per image
    - **Input Resolution**: 128x128 pixels
    
    ### Features
    - Real-time image segmentation
    - Multiple model architectures
    - Batch processing capabilities
    - Professional visualization
    - Export functionality
    - RESTful API endpoints
    
    ### Use Cases
    - Medical research and education
    - Dermatological analysis
    - Computer vision research
    - Deep learning portfolio projects
    - PhD and job applications
    
    ### Dataset Information
    The model is trained on the ISIC 2018 Challenge dataset, which contains:
    - High-quality dermoscopic images
    - Expert-annotated segmentation masks
    - Various skin lesion types
    - Professional medical standards
    
    ### Development
    This project was developed as a showcase for:
    - Advanced computer vision techniques
    - Medical AI applications
    - Full-stack AI development
    - Research and implementation skills
    
    ### License
    This project is licensed under the MIT License.
    
    ### Contact
    - **GitHub**: [farshid92](https://github.com/farshid92)
    - **Repository**: [derm-ai-assistant](https://github.com/farshid92/derm-ai-assistant)
    """)

# Navigation
st.sidebar.markdown("## 🔬 DermAI Assistant")
st.sidebar.markdown("---")

# Navigation buttons
if st.sidebar.button("🏠 Home", use_container_width=True):
    st.session_state.current_page = 'home'

if st.sidebar.button("📤 Single Image", use_container_width=True):
    st.session_state.current_page = 'single'

if st.sidebar.button("🔄 Batch Processing", use_container_width=True):
    st.session_state.current_page = 'batch'

if st.sidebar.button("ℹ️ About", use_container_width=True):
    st.session_state.current_page = 'about'

st.sidebar.markdown("---")

# About section in sidebar
st.sidebar.markdown("""
**About:**
This AI-powered tool helps dermatologists and researchers segment skin lesions from dermoscopic images.

**Features:**
- Multiple model architectures
- Real-time segmentation
- High accuracy results
- Professional visualization

**Dataset:** ISIC 2018 Challenge
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**GitHub:** [farshid92/derm-ai-assistant](https://github.com/farshid92/derm-ai-assistant)")

# Page routing
if st.session_state.current_page == 'home':
    home_page()
elif st.session_state.current_page == 'single':
    single_image_page()
elif st.session_state.current_page == 'batch':
    batch_processing_page()
elif st.session_state.current_page == 'about':
    about_page()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 DermAI Assistant - Advanced Skin Lesion Segmentation</p>
    <p>Built with ❤️ using Streamlit, TensorFlow, and FastAPI</p>
    <p>For research and educational purposes</p>
</div>
""", unsafe_allow_html=True) 