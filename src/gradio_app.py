import gradio as gr
from PIL import Image, ImageFilter
import numpy as np

def rotate_image(image_path, degrees=270):
    """
    Rotates the image clockwise. 
    The uploaded image is 90 degrees counter-clockwise, 
    so rotating 270 degrees counter-clockwise (or 90 clockwise) fixes it.
    """
    print(f"Rotating image by {degrees} degrees...")
    with Image.open(image_path) as img:
        return img.convert("L").rotate(degrees, expand=True)

# examples
# https://huggingface.co/spaces/Gradio-Blocks/uniformer_image_segmentation/blob/main/app.py


# def greet(name, intensity):
#     return "Hello, " + name + "!" * int(intensity)

# demo = gr.Interface(
#     fn=greet,
#     inputs=["text", "slider"],
#     outputs=["text"],
#     api_name="predict"
# )

# demo = gr.Interface(
#     fn=rotate_image, 
#     inputs=[gr.Image(type="filepath")], 
#     outputs=gr.Image()
# )

# def update(name):
#     return f"Welcome to Gradio, {name}!"

# with gr.TabbedInterface(interface_list) as demo:

def process_image_edits(image_editor_dict):
    """
    Process the dictionary returned by gr.ImageEditor.
    'image_editor_dict' contains:
    - 'background': The original uploaded image.
    - 'layers': Any drawings or masks made by the user.
    - 'composite': The final image with crops, rotations, and layers applied.
    """
    if image_editor_dict is None:
        return None
    
    # Extract the composite image (which includes the user's crop and rotation)
    composite_img = image_editor_dict.get("composite")
    
    if composite_img is not None:
        # composite_img is a PIL Image object
        # You can perform further AI or image processing here
        return composite_img
    
    # Fallback to background if composite is empty
    return image_editor_dict.get("background")


# with gr.Blocks() as demo:
#     gr.Markdown("Select an image file.")
#     with gr.Row():
#         inp = gr.Image(type="filepath"),
#         out = gr.Image()
#     btn = gr.Button("Run")
#     btn.click(fn=rotate_image, inputs=inp, outputs=out)

with gr.Blocks() as demo:
    gr.Markdown("# Image Editor: Rotate and Crop Example")
    gr.Markdown("Upload an image, use the crop and rotate tools, then click 'Apply Edits'.")
    
    with gr.Row():
        inp_img = gr.Image(type="filepath"),
        out_img = gr.Image()
    btn = gr.Button("Rotate")
    btn.click(fn=rotate_image, inputs=inp_img, outputs=out_img)
    with gr.Row():
        with gr.Column():
            # gr.ImageEditor enables native crop/rotate/draw
            input_editor = gr.ImageEditor(
                {"background": out_img, "layers": None},
                label="Edit Your Image",
                type="pil",
                transforms=["crop", "rotate"], # Limit tools to cropping and rotation
                sources=["upload"]
            )
            crop_btn = gr.Button("Crop")
            
        with gr.Column():
            output_image = gr.Image(label="Processed Result")

    # Link the button to the function
    submit_btn.click(
        fn=process_image_edits,
        inputs=input_editor,
        outputs=output_image
    )


# with gr.Blocks() as demo:
#     input_text = gr.Textbox()

#     @gr.render(inputs=input_text)
#     def show_split(text):
#         if len(text) == 0:
#             gr.Markdown("## No Input Provided")
#         else:
#             for letter in text:
#                 with gr.Row():
#                     text = gr.Textbox(letter)
#                     btn = gr.Button("Clear")
#                     btn.click(lambda: gr.Textbox(value=""), None, text)

# with gr.Blocks() as demo:
#     button = gr.Button(label="Generate Image")
#     button.click(fn=image_generator, inputs=gr.Textbox(), outputs=gr.Image())
# demo.queue(max_size=10)
# demo.launch()

# with gr.Blocks() as demo:
#     with gr.Row():
#         with gr.Column(scale=1):
#             text1 = gr.Textbox()
#             text2 = gr.Textbox()
#         with gr.Column(scale=4):
#             btn1 = gr.Button("Button 1")
#             btn2 = gr.Button("Button 2")


# with gr.Blocks() as demo:
#     track_count = gr.State(1)
#     add_track_btn = gr.Button("Add Track")

#     add_track_btn.click(lambda count: count + 1, track_count, track_count)

#     @gr.render(inputs=track_count)
#     def render_tracks(count):
#         audios = []
#         volumes = []
#         with gr.Row():
#             for i in range(count):
#                 with gr.Column(variant="panel", min_width=200):
#                     gr.Textbox(placeholder="Track Name", key=f"name-{i}", show_label=False)
#                     track_audio = gr.Audio(label=f"Track {i}", key=f"track-{i}")
#                     track_volume = gr.Slider(0, 100, value=100, label="Volume", key=f"volume-{i}")
#                     audios.append(track_audio)
#                     volumes.append(track_volume)

#             def merge(data):
#                 sr, output = None, None
#                 for audio, volume in zip(audios, volumes):
#                     sr, audio_val = data[audio]
#                     volume_val = data[volume]
#                     final_track = audio_val * (volume_val / 100)
#                     if output is None:
#                         output = final_track
#                     else:
#                         min_shape = tuple(min(s1, s2) for s1, s2 in zip(output.shape, final_track.shape))
#                         trimmed_output = output[:min_shape[0], ...][:, :min_shape[1], ...] if output.ndim > 1 else output[:min_shape[0]]
#                         trimmed_final = final_track[:min_shape[0], ...][:, :min_shape[1], ...] if final_track.ndim > 1 else final_track[:min_shape[0]]
#                         output += trimmed_output + trimmed_final
#                 return (sr, output)

#             merge_btn.click(merge, set(audios + volumes), output_audio)

#     merge_btn = gr.Button("Merge Tracks")
#     output_audio = gr.Audio(label="Output", interactive=False)


demo.launch()