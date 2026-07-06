# from openai.types import ResponseFormatJSONSchema
# from openai.types.responses import response_text_config_param
# from PIL import Image
import gradio as gr
from dotenv import load_dotenv
from io import BytesIO
import base64
import pandas as pd
import numpy as np
from openai import OpenAI
import ocr_model as ocr
import src.logger_config as logger_config
import logging

# Initialize logger
logger = logger_config.logger

load_dotenv()


# State variables for OCR progress tracking
ocr_progress = gr.State(0)
ocr_status = gr.State("")

APP_HOME = "/Users/jon/Projects/ebird-format"
OUTPUT_PATH = f"{APP_HOME}/survey"
REFERENCE_PATH = f"{APP_HOME}/data-reference"
OCR_MODELS = [
    "mlx-community--olmOCR-2-7B-1025-bf16",
    "olmOCR-2-7B-1025-mlx-8bit",
    "mlx-community--GLM-OCR-bf16",
    "LightOnOCR-2-1B-bf16",  
    "chandra-ocr-2-mxfp8-mlx"
    ]
SURVEY_SITES = [
    "Arboretum", "Carkeek", "Cheasty", "Discovery",
     "Genesee", "Golden Gardens", "Lake Forest Park", 
     "Lincoln Park", "Magnuson", "Seward"
     ]

# Create Gradio app using Blocks for more control
with gr.Blocks() as demo:
    # State tracking variables across tabs
    saved_file_path = gr.State("")
    ocr_progress = gr.State(0)
    ocr_status = gr.State("")

    with gr.Tab("Image Preprocessing"):
        gr.Markdown("## Crop and rotate image files")
        with gr.Row(equal_height=False):
            with gr.Column():
                input_img = gr.ImageEditor(
                    height='80vh', type="filepath", sources="upload",
                    brush=False, eraser=False, transforms=["crop"], label="Workspace (Original image)"
                )
                process_btn = gr.Button("Rotate Image", variant="primary")
            with gr.Column():
                output_img = gr.Image(height='80vh', label="Corrected image", type="pil")
                # zoom_preview = gr.Image(height='40vh', label="Zoom Preview", type="pil")
                # generate_preview_btn = gr.Button("Generate Zoom Preview", variant="secondary")

        with gr.Sidebar():
            angle_slider = gr.Slider(0, 360, value=270, step=0.5, label="Rotation angle")
            site_selector = gr.Dropdown(
                choices=SURVEY_SITES, 
                value=SURVEY_SITES[-1], 
                label="Survey Site", 
                interactive=True)
            page_indicator = gr.Radio(choices=('page1', 'page2'), value='page1', label="Page")
            date_picker = gr.DateTime(include_time=False, label="Survey Date", type="string")
            out_text = gr.Text(label="Saved absolute path:")
            save_btn = gr.Button("Save Image", variant="primary")

        process_btn.click(
            fn=ocr.rotate_image,
            inputs=[input_img, angle_slider],
            outputs=[output_img, input_img]
        )
        
        # generate_preview_btn.click(
        #     fn=lambda img_path: Image.open(img_path).convert("L") if img_path else None,
        #     inputs=[input_img],
        #     outputs=[zoom_preview]
        # )

        save_btn.click(
            fn=ocr.save_rotated_image,
            inputs=[date_picker, page_indicator, output_img], 
            outputs=[out_text]
        ).then(
            fn=lambda path: path if path and not path.startswith("Error") else "",
            inputs=[out_text],
            outputs=[saved_file_path]
        )

    with gr.Tab("OCR Text Extraction"):
        gr.Markdown("## Extract Text from Images")
        with gr.Row(equal_height=False):
            ocr_p1 = gr.Image(
                height='50vh', 
                label="OCR input image target", 
                interactive=True)
            ocr_p2 = gr.Image(
                height='50vh', 
                label="OCR input image target", 
                interactive=True)
        with gr.Row(equal_height=False):
            file_picker = gr.File(label="OCR JSON results:", interactive=True, file_types=[".json"])

        with gr.Row(equal_height=False):
            output_frame = gr.DataFrame(
                value=ocr.load_multiple_files, 
                inputs=file_picker, 
                interactive=True, wrap=True)

        with gr.Sidebar():
            model_selector = gr.Dropdown(
                choices=ocr.OCR_MODELS, 
                value=ocr.OCR_MODELS[0], 
                label="OCR Model", 
                interactive=True)
            run_ocr_btn = gr.Button("Run OCR Extraction", variant="primary", interactive=True)
            stop_btn = gr.Button("Stop OCR", variant="stop")
            progress_bar = gr.Slider(0, 100, value=0, label="Progress (%)", interactive=False)
            status_text = gr.Textbox(label="Status", placeholder="Ready to start...", interactive=False)
            frame_path = gr.Text(label="OCR output file:", interactive=True)
            output_header = gr.Text(label="Page Header", interactive=True)
            csv_filename = gr.Text(label="Validated OCR data saved as:")
            csv_btn = gr.Button("Save Validated Results", variant="primary")
            report_btn = gr.Button("Generate Formatted Report", variant="primary")

        with gr.Row(equal_height=False):
            report_csv = gr.File(
                label="Validated OCR results:", 
                interactive=True, 
                file_types=[".csv"])
        with gr.Row(equal_height=False):
            report_df = gr.DataFrame(
                value=ocr.format_results,
                inputs=report_csv,
                label="Final formatted report:", 
                interactive=True)

        def update_display(status, progress):
            """Update display with status and progress."""
            status_text = status
            progress_bar = progress
            return status_text, progress_bar

        run_ocr_btn.click(
            fn=lambda: ocr.set_ocr_stopped(False),
            inputs=[],
            outputs=[]
        ).then(
            fn=ocr.run_ocr_pipeline,
            inputs=[model_selector, date_picker, ocr_p1, ocr_p2],
            outputs=[output_header, frame_path],
            show_progress="full",
            api_name="ocr_extract"
        ).then(
            fn=update_display,
            inputs=[ocr_status, ocr_progress],
            outputs=[status_text, progress_bar]
        )

        csv_btn.click(
            fn=ocr.df_to_csv, 
            inputs=[output_frame, frame_path],
            outputs=csv_filename)
        
        report_btn.click(
            fn=ocr.format_results,
            inputs=report_csv,
            outputs=report_df)

        # Stop OCR button handler
        stop_btn.click(
            fn=lambda: ocr.set_ocr_stopped(True),
            inputs=[],
            outputs=[]
        )


if __name__ == "__main__":
    valid = [APP_HOME, OUTPUT_PATH, REFERENCE_PATH, f"{APP_HOME}/src"]
    demo.launch(allowed_paths=valid, share=True)
