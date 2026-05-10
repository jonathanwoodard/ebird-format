import os
import json
import pandas as pd
from PIL import Image
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template

# 1. Configuration
cwd = os.getcwd()
parent_dir = cwd.rsplit('/', 1)[0]
MODEL_PATH = "mlx-community/gemma-4-26b-a4b-mxfp4"
IMAGE_FILE = "/Users/jon/Downloads/example.jpg"

def setup_data():
    """Load reference species codes from CSVs."""
    most_likely = pd.read_csv(f'{parent_dir}/data-raw/most_likely.csv')['Code'].tolist()
    aos_full = set(pd.read_csv(f'{parent_dir}/data-raw/aos_full.csv')['Code'].tolist())
    return most_likely, aos_full

def split_notebook_image(image_path):
    """Splits a 2-page spread into Left and Right pages."""
    img = Image.open(image_path)
    width, height = img.size
    mid = width // 2
    
    left_page = img.crop((0, 0, mid, height))
    right_page = img.crop((mid, 0, width, height))
    
    left_path, right_path = "left_page.jpg", "right_page.jpg"
    left_page.save(left_path)
    right_page.save(right_path)
    return left_path, right_path

def extract_from_page(model, processor, config, image_path, most_likely_codes):
    """Processes a single page using Gemma 4 with contextual hints."""
    # Convert list to a string for the prompt
    likely_str = ", ".join(most_likely_codes)
    
    prompt = f"""<|user|>
        <|image|>
        You are an expert ornithologist transcribing field notes.
        The image is a page from a birding notebook. 
        Format: [Species Code] | [Count 1] | [Count 2]. 
        If a column is empty or has a '-', use null.

        Context: 
        - Species codes are 4 letters. 
        - Most likely codes in this area: {likely_str}
        - If handwriting is messy, prioritize matching to the codes above.

        Return a JSON object with:
        "header": "text from top of page",
        "groups": [
        {{"time": "time value", "records": [{{"species": "CODE", "c1": val, "c2": val}}]}}
        ]
        <|assistant|>
        """
    
    image = [Image.open(image_path)]

    # Apply chat template
    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images=len(image)
    )

    output = generate(model, processor, formatted_prompt, image, max_tokens=2048, temp=0.0, verbose=False)
    return output

def validate_codes(json_data, aos_full):
    """Checks extracted codes against the full AOS list."""
    try:
        data = json.loads(json_data)
        for group in data.get("groups", []):
            for record in group.get("records", []):
                code = record['species']
                if code not in aos_full:
                    record['valid_code'] = False
                    print(f"Warning: Code '{code}' not found in AOS master list.")
                else:
                    record['valid_code'] = True
        return data
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return json_data

# Main Execution
if __name__ == "__main__":
    # Load model and data
    most_likely, aos_full = setup_data()
    model, processor = load(MODEL_PATH)
    config = load_config(MODEL_PATH)

    # Pre-process images
    pages = split_notebook_image(IMAGE_FILE)
    
    results = {}
    for page_path in pages:
        print(f"Processing {page_path}...")
        raw_json = extract_from_page(model, processor, page_path, most_likely)
        structured_data = validate_codes(raw_json, aos_full)
        results[page_path] = structured_data

    # Save final output
    with open("final_observations.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Extraction complete. Results saved to final_observations.json")
