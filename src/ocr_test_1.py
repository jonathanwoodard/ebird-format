import argparse
import os
import re
import json
import pandas as pd
from PIL import Image
# import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template
# from mlx_vlm.generate import stream_generate
from ocr_preprocess import *


cwd = os.getcwd()
parent_dir = cwd.rsplit('/src', 1)[0]
# MODEL_PATH = "alexgusevski/olmOCR-7B-0225-preview-q4-mlx"
MODEL_PATH = "mlx-community/olmOCR-2-7B-1025-bf16"
# MODEL_PATH = "mlx-community/PaddleOCR-VL-1.5-bf16"
# MODEL_PATH = "mlx-community/GLM-OCR-bf16"
# MODEL_PATH = "mlx-community/MinerU2.5-2509-1.2B-bf16"
IMAGE_FILE1 = f"{cwd}/output_segments/left_page_final.jpg"
IMAGE_FILE2 = f"{cwd}/output_segments/right_page_final.jpg"

def setup_data():
    """Load reference species codes from CSVs."""
    most_likely = pd.read_csv(f'{parent_dir}/data-raw/most_likely.csv')['Code'].tolist()
    aos_full = set(pd.read_csv(f'{parent_dir}/data-raw/aos_full.csv')['Code'].tolist())
    return most_likely, aos_full

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def search_pool(candidates, target_pool):
    """
    Helper function to find the closest match inside a specific pool of codes.
    Returns (best_match_string, minimum_distance_found)
    """
    best_match = None
    min_distance = float('inf')
    
    for guess in candidates:
        guess_clean = str(guess).upper().strip()
        if not guess_clean:
            continue
            
        if guess_clean in target_pool:
            return guess_clean, 0  # Perfect match found, immediate success
            
        for valid_code in target_pool:
            dist = levenshtein_distance(guess_clean, valid_code)
            if dist < min_distance:
                min_distance = dist
                best_match = valid_code
                
    return best_match, min_distance

def match_code_to_tiered_lexicon(guesses, likely_codes, unlikely_codes):
    """
    Evaluates alternative guesses against a tiered lexicon.
    Checks the 'likely' group first, falling back to 'unlikely' if needed.
    """
    likely_upper = {code.upper() for code in likely_codes}
    unlikely_upper = {code.upper() for code in unlikely_codes}
    
    # Tier 1: Look inside the likely codes pool
    best_likely_match, likely_dist = search_pool(guesses, likely_upper)
    
    # If we found a confident match in the likely pool, use it
    if likely_dist <= 2:
        return best_likely_match
        
    # Tier 2 Fallback: If no good likely match was found, search the unlikely pool
    best_unlikely_match, unlikely_dist = search_pool(guesses, unlikely_upper)
    
    if unlikely_dist <= 2:
        print(f"-> Falling back to unlikely pool match: {best_unlikely_match} (distance: {unlikely_dist})")
        return best_unlikely_match
        
    # Final Tie-Breaker/Safety: If both are poor matches (>2 distance), 
    # take whichever one was structurally closer overall
    if likely_dist <= unlikely_dist and best_likely_match:
        return best_likely_match
    elif best_unlikely_match:
        return best_unlikely_match
        
    return None

def extract_json_from_string(raw_response):
    """
    Extracts and parses JSON from text. If the model outputs single quotes 
    instead of double quotes, this automatically sanitizes the string.
    """
    try:
        text = str(raw_response.text).strip()
        
        # 1. Isolate the JSON chunk using markdown indicators or outer curly braces
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        json_str = match.group(1) if match else text
        
        json_str = json_str.strip()
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            print(f"Error: No valid JSON bounding brackets found in text:\n{text}")
            return None
            
        json_clean = json_str[start_idx:end_idx]
        return json.loads(json_clean)
        
    except Exception as e:
        print(f"Failed to extract JSON from raw text. Error Details: {e}")
        print(f"Cleaned text string trying to parse was:\n{json_clean if 'json_clean' in locals() else text}")
        raise

def process_page_with_mlx_vlm(model, processor, config, image_obj):
    """
    Corrected MLX-VLM execution loop using proper keyword arguments 
    and prompt formatting templates.
    """
    print("Sending image to mlx-vlm for native Apple Silicon inference...")
    
    raw_prompt = """
    You are an expert ornithologist transcribing field notes from a survey.
    The image is a handwritten page from a birding notebook. Pages may include a header 
    with date (month/day) and a brief description of weather (temperature, wind, 
    precipitation, cloud cover).
    Each page will have four data blocks with a timestamp (hours:minutes) in 
    the upper left. These sections may also be labelled with a site id (e.g. #1, #8)
    The data blocks will be located upper left, upper right, lower left, lower right.
    Tabular format: [four letter code] | [Count 1] | [Count 2]. 
    A typical row of data will look like: "BEWR    -    3"  and each data block may contain 
    between one and 25 rows of data. Most data blocks will contain between five and ten rows.
    Analyze the tabular content in this document and extract the data column-by-column. 
    The first column will always contain four capitalized letters and will never contain numbers or special characters. 
    The second and third columns are counts and will contain either integers, white space or "-". At least one of these
    columns MUST contain an integer count value.

    You must return your response inside a valid JSON markdown block exactly like this, 
    with one 'data_block n' section for each of the four blocks where n is the number of the block:
    ```json
    {"header": {"type": "string"},
    "data_block n": {
    "start_time": {"type": "string"},
    "records": [
    {"primary_code_guess": {"type": "string"},
    "seen": {"type": "string"},
    "heard": {"type": "string"}}]}}
    ```
    """

    # 1. Structure the prompt with the model's required formatting template tokens
    formatted_prompt = apply_chat_template(
        processor,
        config,
        raw_prompt,
        num_images=1
    )
    
    # 2. Execute with explicit keyword targeting to prevent argument swapping
    raw_response = generate(
        model=model, 
        processor=processor, 
        image=image_obj, 
        prompt=formatted_prompt, 
        max_tokens=2048, 
        verbose=False
    )
    return raw_response

def main(model_path, img_path, output_path):
    """
    perform image preprocessing and ocr
    """
    parser = argparse.ArgumentParser(description="Extract handwritten records from notebook page images")
    parser.add_argument("--model", "-m", required=True, help="OCR model name (hugingface format)")
    parser.add_argument("--path", "-p", type=str, required=True, help="Input image file path")
    parser.add_argument("--filename", "-f", type=str, required=True, help="Image file name")
    parser.add_argument("--out", "-o", type=str, default=None, help="Output folder (default: input folder)")
    args = parser.parse_args()

    most_likely, aos_full = setup_data() 
    model, processor = load(model_path)
    config = load_config(model_path)
    img_obj = Image.open(img_path)

    raw_response = process_page_with_mlx_vlm(
                    model=model, 
                    processor=processor, 
                    config=config, 
                    image_obj=img_obj
                )
    
    try:
        extracted_data = extract_json_from_string(raw_response)
        for n in range(1, 5):
            raw_ocr = extracted_data[f'data_block {n}']
            # for now, only checking primary guess
            for record in raw_ocr['records']:
                code = record['primary_code_guess']
                record['validated_code'] = match_code_to_tiered_lexicon([code], most_likely, aos_full)
        return extracted_data
    except Exception as e:
        return raw_response


# Main Execution
if __name__ == "__main__":
    # define model and file path options
    cwd = os.getcwd()
    parent_dir = cwd.rsplit('/src', 1)[0]
    # MODEL_PATH = "alexgusevski/olmOCR-7B-0225-preview-q4-mlx"
    # MODEL_PATH = "mlx-community/olmOCR-2-7B-1025-bf16" very good
    # MODEL_PATH = "mlx-community/PaddleOCR-VL-1.5-bf16"
    # MODEL_PATH = "mlx-community/GLM-OCR-bf16"
    # MODEL_PATH = "mlx-community/MinerU2.5-2509-1.2B-bf16" poor
    IMAGE_FILE1 = f"{cwd}/output_segments/left_page_final.jpg"
    IMAGE_FILE2 = f"{cwd}/output_segments/right_page_final.jpg"
    IMAGE_FILE3 = f"{cwd}/output_segments/right_page_segl.jpg"

    # Pre-process images
    # set up argparse for ocr_preprocess.py
    
