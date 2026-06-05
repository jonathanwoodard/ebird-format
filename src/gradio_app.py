"""
Gradio app for handwriting OCR with local MLX-VLM model.

Images undergo preprocessing (rotation, deskew, boundary crop, split)
before being fed to the OCR model.
"""

import gradio as gr
from PIL import Image
from ocr_preprocess import pipeline
from ocr_test_1 import run_ocr_inference
import os


def preprocess_and_ocr(image_path):
    """
    Pipeline: rotate -> deskew -> crop -> split -> OCR -> show both pages.
    """
    try:
        # Run preprocessing pipeline
        pipeline(image_path)

        # Get output directory from pipeline logs
        # The pipeline saves to "output_segments/" by default
        output_dir = "output_segments"

        # Check if output exists, otherwise use default
        if not os.path.exists(output_dir):
            output_dir = "output"

        # Get page paths (try both conventions used in pipeline)
        left_page_path = None
        right_page_path = None

        # Try full paths first
        if os.path.exists(os.path.join(output_dir, "left_page_final.jpg")):
            left_page_path = os.path.join(output_dir, "left_page_final.jpg")
        if os.path.exists(os.path.join(output_dir, "right_page_final.jpg")):
            right_page_path = os.path.join(output_dir, "right_page_final.jpg")

        # Try alternative naming
        if not left_page_path:
            left_page_path = os.path.join(output_dir, "left_page.jpg")
        if not right_page_path:
            right_page_path = os.path.join(output_dir, "right_page.jpg")

        # Run OCR on combined image
        if left_page_path and right_page_path:
            # Combine pages for OCR
            combined_img = Image.open(left_page_path).convert("RGB")
            combined_img = Image.open(right_page_path).convert("RGB")
            combined_img = Image.new("RGB", (combined_img.width, combined_img.height), combined_img)
            combined_img.paste(combined_img, (0, 0))

            ocr_result = run_ocr_inference(combined_img)

            return (
                ocr_result.get("text", "No text extracted"),
                left_page_path,
                right_page_path
            )

        return (
            "Could not locate preprocessed pages. Check 'output_segments/' directory.",
            None,
            None
        )

    except Exception as e:
        return (
            f"Error: {str(e)}",
            None,
            None
        )


def run_ocr_from_file(image_path):
    """
    OCR directly from uploaded image (bypass preprocessing).
    """
    try:
        img = Image.open(image_path).convert("RGB")
        ocr_result = run_ocr_inference(img)
        return ocr_result.get("text", "No text extracted"), None, None
    except Exception as e:
        return f"Error: {str(e)}", None, None


def load_sample_image():
    """
    Load a sample image for the user to try immediately.
    """
    sample_dir = os.path.join(os.path.dirname(__file__), "sample-images")
    sample_file = os.path.join(sample_dir, "sample.jpg")

    if os.path.exists(sample_file):
        return sample_file, None, None
    return None, None, None


with gr.Blocks(title="Handwriting OCR") as demo:
    gr.Markdown("""
    # Handwriting OCR with MLX-VLM

    Process handwritten field notes with local OCR. The pipeline includes:
    - Rotation correction (90° clockwise)
    - Deskew detection and correction
    - Boundary crop and page split
    - OCR inference with olmOCR-2-7B model
    """)

    gr.Markdown("""
    ## Preprocessing Pipeline

    1. **Rotate**: Fixes 90° counter-clockwise upload orientation
    2. **Deskew**: Detects and corrects minor alignment skew
    3. **Crop**: Tightly crops to notebook edges
    4. **Split**: Separates left/right pages
    5. **OCR**: Extracts text with MLX-VLM model
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload")
            input_image = gr.Image(
                label="Upload Image",
                type="filepath",
                sources=["upload", "clipboard"],
                height=300
            )

            preprocess_btn = gr.Button("Process", variant="primary", value="Process Image", size="lg")

            gr.Markdown("### Preprocessed Pages (for reference)")
            left_page_display = gr.Image(label="Left Page", interactive=False)
            right_page_display = gr.Image(label="Right Page", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### OCR Result")
            ocr_output = gr.Textbox(
                label="Extracted Text",
                lines=8,
                max_lines=10,
                interactive=True,
                placeholder="OCR output will appear here..."
            )

    preprocess_btn.click(
        fn=preprocess_and_ocr,
        inputs=input_image,
        outputs=[ocr_output, left_page_display, right_page_display]
    )

    gr.Markdown("---")
    gr.Markdown("## Quick OCR (no preprocessing)")

    quick_ocr_img = gr.Image(
        label="Quick OCR",
        type="filepath",
        sources=["upload", "clipboard"],
        height=300
    )

    quick_btn = gr.Button("Quick OCR", variant="secondary")

    quick_btn.click(
        fn=run_ocr_from_file,
        inputs=quick_ocr_img,
        outputs=[ocr_output, left_page_display, right_page_display]
    )

    gr.Markdown("---")
    gr.Markdown("## Sample Image")

    sample_input, sample_left, sample_right = load_sample_image()

    if sample_input:
        sample_btn = gr.Button("Load Sample Image", variant="secondary")
        sample_btn.click(
            fn=lambda p: p,
            inputs=sample_input,
            outputs=input_image
        )
        sample_left_display = gr.Image(label="Sample Left", interactive=False, value=sample_left)
        sample_right_display = gr.Image(label="Sample Right", interactive=False, value=sample_right)
        sample_ocr = gr.Textbox(label="Sample Result", lines=8, interactive=True)

        sample_btn.click(
            fn=preprocess_and_ocr,
            inputs=sample_input,
            outputs=[sample_ocr, sample_left_display, sample_right_display]
        )


demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)
