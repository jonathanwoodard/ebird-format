import argparse
import os
import re
import json
import pandas as pd
import numpy as np
from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template
from ocr_preprocess import *
import gradio as gr
from openai import OpenAI

SPECIES = "/Users/jon/Projects/ebird-format/data-raw/IBP-AOS-LIST24.csv.csv"

def rotate_image(editor_dict, degrees=270):
    """
    ImageEditor returns a dict. Use 'composite' value
    Rotates the image clockwise. 
    The uploaded image is 90 degrees counter-clockwise, 
    so rotating 270 degrees counter-clockwise (or 90 clockwise) fixes it.
    """
    print(f"Rotating image by {degrees} degrees...")
    with Image.open(editor_dict["composite"]) as img:
        return img.convert("L").rotate(degrees, expand=True), img

def codes2names():
    band_codes = pd.read_csv(SPECIES)
    cols = ["SPEC", "COMMONNAME", "SCINAME"]
    band_codes = band_codes[cols]
    band_codes["Genus"] = band_codes.SCINAME.apply(lambda x: x.split(" ")[0])
    band_codes["Species"] = band_codes.SCINAME.apply(lambda x: x.split(" ")[1])
    cols = ["SPEC", "COMMONNAME", "Genus", "Species"]
    band_codes = band_codes[cols]
    band_codes.columns = ["Code", "Common Name", "Genus", "Species"]
    return band_codes

def format_results(ocr_results):
    """
    Format validated OCR results for ebird .csv upload
    """

    # Columns for .csv upload in correct order
    report_cols = ["Common Name", "Genus", "Species", "Number", "Species Comments", 
        "Location Name", "Latitude", "Longitude", "Date", "Start Time", 
        "State/Province", "Country Code", "Protocol", "Number of Observers", 
        "Duration", "All observations reported?", "Effort Distance Miles", 
        "Effort area acres", "Submission Comments"]

    # Get mapping of band codes to species names
    band_codes = codes2names()
    df = pd.read_csv(ocr_results)
    df['Seen'] = df.Seen.apply(lambda x: int(x) if x!="-" else 0)
    df['Heard'] = df.Heard.apply(lambda x: int(x) if x!="-" else 0)
    df["Number"] = df[["Seen", "Heard"]].sum(axis=1)
    df['Seen'] = df.Seen.apply(lambda x: f"{x} heard" if x!=0 else None)
    df['Heard'] = df.Heard.apply(lambda x: f"{x} heard" if x!=0 else None)

    # Combine Seen/Heard counts to species record comment
    _comments = df[["Seen", "Heard"]].values
    species_comments = []
    for c in _comments:
        vals = [x for x in c if type(x)==str]
        species_comments.append(", ".join(vals))
    df["Species Comments"] = species_comments

    # Format timestamps
    times = pd.DataFrame(df["Start Time"].sort_values().drop_duplicates().reset_index(drop=True))
    times["Location Name"] = ROUTES["Seward West"][:len(times)]
    times[["Latitude", "Longitude", ]] = [None, None]
    # times["Date"] = "05/09/2026" # date from datepicker
    times['start_time'] = pd.to_datetime(times["Start Time"], format="%H:%M").dt.strftime("%H:%M")
    times[report_cols[10:]] = ["WA", "US", "stationary", "1", "5", "yes", None, None, "weather"]

    # Order records and survey stations by survey start time
    _df = df.merge(times, on=["Start Time"])
    _df = _df[_df.columns[6:]]
    _df["Start Time"] = _df.start_time.copy()
    result = band_codes.merge(_df, left_on="Code", right_on="Validated Code")
    return result[report_cols]

def setup_data(_most_likely, _aos_full):
    """Load reference species codes from CSVs."""
    most_likely = pd.read_csv(_most_likely)['Code'].tolist()
    aos_full = set(pd.read_csv(_aos_full)['Code'].tolist())
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
    Currently, the OCR model is only returning a single guess (no alternates)
    Returns (best_match_string, minimum_distance_found)
    """
    
    matches = {}
    for guess in candidates:
        guess_clean = str(guess).upper().strip()
        if not guess_clean:
            continue
            
        if guess_clean in target_pool:
            matches[0] = [guess_clean]
            return matches  # Perfect match found, immediate success
            
        matches[1] = []
        matches[2] = []
        for valid_code in target_pool:
            dist = levenshtein_distance(guess_clean, valid_code)
            # return ALL matches at a distance of 1 or 2 (limit 10)
            if dist == 1: 
                matches[1].append(valid_code)
                if len(matches[1]) > 10:
                    break
            elif dist == 2:
                matches[2].append(valid_code)
                if len(matches[2]) > 10:
                    break
            else:
                continue
        if len(matches[1]) > 0:
            del (matches[2])
            return matches
        elif len(matches[2]) > 0:
            del (matches[1])
            return matches
        else:
            return None

def match_code_to_tiered_lexicon(guesses, likely_codes, unlikely_codes):
    """
    Evaluates alternative guesses against a tiered lexicon.
    Checks the 'likely' group first, falling back to 'unlikely' if needed.
    """
    likely_upper = {code.upper() for code in likely_codes}
    unlikely_upper = {code.upper() for code in unlikely_codes}
    
    # Tier 1: Look inside the likely codes pool
    likely_matches = search_pool(guesses, likely_upper)
    # If we found a confident match in the likely pool, use it
    if likely_matches is not None:
        likely_dist = np.min(list(likely_matches.keys()))
        if likely_dist > 0:
            best_likely_matches = ', '.join(likely_matches[likely_dist])
            print(f"-> No exact matches found. Best matches: {best_likely_matches} (distance: {likely_dist})")
        return likely_matches
        
    # Tier 2 Fallback: If no good likely match was found, search the unlikely pool
    unlikely_matches = search_pool(guesses, unlikely_upper)
    
    if unlikely_matches is not None:
        unlikely_dist = np.min(list(unlikely_matches.keys()))
        best_unlikely_matches = ', '.join(unlikely_matches[unlikely_dist])
        print(f"-> Falling back to unlikely pool match: {best_unlikely_matches} (distance: {unlikely_dist})")
        return unlikely_matches
        
    # If both are poor matches (>2 distance), return original guesses
    # take whichever one was structurally closer overall
    else:
        print(f"-> No matches for {guesses} found with distance <= 2; manual result validation required")
        return None

def extract_json_from_string(raw_response):
    """
    Extracts and parses JSON from text. If the model outputs single quotes 
    instead of double quotes, this automatically sanitizes the string.
    """
    try:
        # text = str(raw_response.text).strip()
        text = str(raw_response).strip()
        
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

# def img_resize(image_obj):
#     """
#     Helper function to resize images - max image dimension should be 1288
#     """
#     img_size = image_obj.size
#     img_ratio = 1288.0/np.max(img_size)
#     new_size = tuple([int(np.round(s*img_ratio, 0)) for s in img_size])
#     img_obj2 = image_obj.resize(new_size, resample=Image.LANCZOS)
#     return img_obj2

def encode_image_to_base64(image_path):
    """Encodes an image to base64 for API transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_text(model_path, img_path):
    """
    perform image preprocessing and ocr
    """
    if not img_path or not os.path.exists(img_path):
        return pd.DataFrame(columns=["Block", "Start Time", "Primary Guess", "Seen", "Heard", "Validated Code"]), {"error": "Invalid file path"}

    most_likely, aos_full = setup_data() 
    model, processor = load(model_path)
    config = load_config(model_path)
    img_obj = encode_image_to_base64(img_path)
    # img_obj2 = img_resize(img_obj)

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
        msg = 'OCR Extraction and Validation Complete'
    except Exception as e:
        extracted_data = vars(raw_response) # transform raw model result to dict
        msg = e
    # with open(output_file, 'w') as f:
    #     json.dump(extracted_data, f)
    print(msg)
    return extracted_data
    


