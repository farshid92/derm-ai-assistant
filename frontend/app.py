import streamlit as st
import requests
import io
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import time

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
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🔬 DermAI Assistant")
    st.markdown("---")
    
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
    
    st.markdown("### About")
    st.markdown("""
    This AI-powered tool helps dermatologists and researchers segment skin lesions from dermoscopic images.
    
    **Features:**
    - Multiple model architectures
    - Real-time segmentation
    - High accuracy results
    - Professional visualization
    
    **Dataset:** ISIC 2018 Challenge
    """)
    
    st.markdown("---")
    st.markdown("**GitHub:** [farshid92/derm-ai-assistant](https://github.com/farshid92/derm-ai-assistant)")

# Main content
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

# File upload section
st.markdown("## 📤 Upload Your Dermoscopic Image")
st.markdown("Upload a skin lesion image for AI-powered segmentation analysis.")

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
                    original_array = np.array(original_image.resize((128, 128)))
                    mask_resized = np.array(mask_image.resize((128, 128)))
                    
                    # Create overlay
                    overlay = original_array.copy()
                    if len(mask_resized.shape) == 3:
                        mask_binary = mask_resized[:, :, 0] > 0
                    else:
                        mask_binary = mask_resized > 0
                    
                    overlay[mask_binary] = [255, 0, 0]  # Red overlay for lesion
                    
                    # Create subplot
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=("Original", "Segmentation Mask", "Overlay"),
                        specs=[[{"type": "image"}, {"type": "image"}, {"type": "image"}]]
                    )
                    
                    # Add images to subplot
                    fig.add_trace(go.Image(z=original_array), row=1, col=1)
                    fig.add_trace(go.Image(z=mask_resized), row=1, col=2)
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
    
    # Sample images section
    st.markdown("## 📋 Sample Images")
    st.markdown("Try with these sample images to see the segmentation in action:")
    
    # Display sample image if available
    try:
        sample_image = Image.open("ISIC_0000001.jpg")
        st.image(sample_image, caption="Sample Dermoscopic Image (ISIC_0000001.jpg)", use_column_width=True)
    except:
        st.info("No sample images found. Upload your own image to test the segmentation.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 DermAI Assistant - Advanced Skin Lesion Segmentation</p>
    <p>Built with ❤️ using Streamlit, TensorFlow, and FastAPI</p>
    <p>For research and educational purposes</p>
</div>
""", unsafe_allow_html=True) 