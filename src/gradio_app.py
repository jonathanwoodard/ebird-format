# from openai.types import ResponseFormatJSONSchema
# from openai.types.responses import response_text_config_param
# from PIL import Image
import gradio as gr
# import os
# import json
# from datetime import datetime as dt
# from io import BytesIO
# import base64
# import pandas as pd
# import numpy as np
# from openai import OpenAI
from dotenv import load_dotenv
# from ocr_test_1 import rotate_image, save_rotated_image, load_multiple_files, run_ocr_pipeline, df_to_csv, format_results
from ocr_test_1 import *
load_dotenv()

APP_HOME = "/Users/jon/Projects/ebird-format"
OUTPUT_PATH = f"{APP_HOME}/survey"
REFERENCE_PATH = f"{APP_HOME}/data-reference"
OCR_MODELS = ["mlx-community--olmOCR-2-7B-1025-bf16",
    "olmOCR-2-7B-1025-mlx-8bit",
    "mlx-community--GLM-OCR-bf16",
    "LightOnOCR-2-1B-bf16",  
    "chandra-ocr-2-mxfp8-mlx"]


# Create Gradio app using Blocks for more control
with gr.Blocks() as demo:
    # State tracking variables across tabs
    saved_file_path = gr.State("")

    with gr.Tab("Image Preprocessing"):
        gr.Markdown("## Crop and rotate image files")
        with gr.Row(equal_height=False):
            with gr.Column():
                input_img = gr.ImageEditor(
                    height='80vh', type="filepath", sources="upload",
                    brush=False, eraser=False, transforms=["crop"], label="Original image"
                )
                process_btn = gr.Button("Rotate Image", variant="primary")
            output_img = gr.Image(height='80vh', label="Corrected image", type="pil")

        with gr.Sidebar():
            angle_slider = gr.Slider(0, 360, value=270, step=0.5, label="Rotation angle")
            page_indicator = gr.Radio(choices=('page1', 'page2'), value='page1', label="Page")
            date_picker = gr.DateTime(include_time=False, label="Survey Date", type="string")
            out_text = gr.Text(label="Saved absolute path:")
            save_btn = gr.Button("Save Image", variant="primary")

        process_btn.click(
            fn=rotate_image,
            inputs=[input_img, angle_slider],
            outputs=[output_img, input_img]
        )
        save_btn.click(
            fn=save_rotated_image,
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
            ocr_p1 = gr.Image(height='50vh', label="OCR input image target", interactive=True)
            ocr_p2 = gr.Image(height='50vh', label="OCR input image target", interactive=True)
        with gr.Row(equal_height=False):
            file_picker = gr.File(label="OCR JSON results:", interactive=True, file_types=[".json"])

        with gr.Row(equal_height=False):
            output_frame = gr.DataFrame(
                value=load_multiple_files, 
                inputs=file_picker, 
                interactive=True, wrap=True)

        with gr.Sidebar():
            model_selector = gr.Dropdown(choices=OCR_MODELS, value=OCR_MODELS[0], label="OCR Model", interactive=True)
            run_ocr_btn = gr.Button("Run OCR Extraction", variant="secondary", interactive=True)
            frame_path = gr.Text(label="OCR output file:")
            output_header = gr.Text(label="Page Header", interactive=True)
            csv_filename = gr.Text(label="Validated OCR data saved as:")
            download_btn = gr.Button("Export to CSV", variant="primary")
            report_btn = gr.Button("Generate Report", variant="primary")

        with gr.Row(equal_height=False):
            report_csv = gr.File(
                label="Validated OCR results:", 
                interactive=True, 
                file_types=[".csv"])
        with gr.Row(equal_height=False):
            report_df = gr.DataFrame(
                inputs=report_csv,
                label="Final formatted report:", 
                interactive=True)

        run_ocr_btn.click(
            fn=run_ocr_pipeline,
            inputs=[model_selector, date_picker, ocr_p1, ocr_p2], 
            outputs=[output_header, frame_path],
            show_progress="full", 
            api_name="ocr_extract"
        )

        download_btn.click(
            fn=df_to_csv, 
            inputs=[output_frame, frame_path],
            outputs=csv_filename)
        
        report_btn.click(
            fn=format_results, 
            inputs=report_csv,
            outputs=report_df)


if __name__ == "__main__":
    valid = [APP_HOME, OUTPUT_PATH, REFERENCE_PATH, f"{APP_HOME}/src"]
    demo.launch(allowed_paths=valid, share=True)
