"""
Batch Processing Script for DermAI Assistant
Process multiple images and generate comprehensive reports
"""

import streamlit as st
import requests
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from pathlib import Path
import time

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
                    original_array = np.array(result['original_image'])
                    mask_array = np.array(result['mask_image'])
                    # If mask is single channel, expand to 3 channels for overlay
                    if len(mask_array.shape) == 2:
                        mask_rgb = np.stack([mask_array]*3, axis=-1)
                    else:
                        mask_rgb = mask_array
                    overlay = original_array.copy()
                    overlay[mask_rgb[..., 0] > 0] = [255, 0, 0]
                    st.image(overlay, caption="Overlay", use_column_width=True)
                
                # Metrics
                st.write(f"**Lesion Area:** {result['lesion_area']:,} pixels")
                st.write(f"**Coverage:** {result['lesion_percentage']:.2f}%")
                st.write(f"**Model:** {result['model_used']}")
    
    # Error summary
    error_results = [r for r in results if r['status'] != 'Success']
    if error_results:
        st.markdown("## ❌ Processing Errors")
        for result in error_results:
            st.error(f"**{result['filename']}:** {result['status']}")

def main():
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
                
                # Export results
                st.markdown("## 💾 Export Results")
                
                # Create CSV report
                if any(r['status'] == 'Success' for r in results):
                    successful_results = [r for r in results if r['status'] == 'Success']
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

if __name__ == "__main__":
    main() 