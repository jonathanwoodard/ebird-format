import os
import json
from datetime import datetime as dt
from io import BytesIO
import base64
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import re
from PIL import Image
load_dotenv()

APP_HOME = "/Users/jon/Projects/ebird-format"
OUTPUT_PATH = f"{APP_HOME}/survey"
REFERENCE_PATH = f"{APP_HOME}/data-reference"
AOS_LIST = f"{REFERENCE_PATH}/aos_full.csv"
NBP_LIST = f"{REFERENCE_PATH}/most_likely.csv"
SPECIES = f"{REFERENCE_PATH}/IBP-AOS-LIST24.csv"

ROUTES = {
    "Seward East": ["Seward East St 1", "Seward East St 2", "Seward East St 3", "Seward East St 4",
    "Seward East St 5", "Seward East St 6", "Seward East St 7", "Seward East St 8"],
    "Seward West": ["Seward West St 1", "Seward West St 2", "Seward West St 3", "Seward West St 4",
    "Seward West St 5", "Seward West St 6", "Seward West St 7", "Seward West St 8"]
}

OCR_MODELS = ["mlx-community--olmOCR-2-7B-1025-bf16",
    "olmOCR-2-7B-1025-mlx-8bit",
    "mlx-community--GLM-OCR-bf16",
    "LightOnOCR-2-1B-bf16",
    "chandra-ocr-2-mxfp8-mlx"]

client = OpenAI(
    base_url=os.getenv("OMLX_URL"),
    api_key=os.getenv("OMLX_API_KEY")
)

# Track if OCR was stopped by user (for session state management)
OCR_STOPPED = False
ocr_progress = 0
ocr_current_page = 0  # 0 = before page 1, 1 = page 1 done, 2 = page 2, 3 = complete
ocr_current_model = 0
ocr_status = "Ready"


def rotate_image(editor_dict, degrees=270):
    """Rotates the image clockwise. The uploaded image is 90 degrees counter-clockwise,
    so rotating 270 degrees counter-clockwise (or 90 clockwise) fixes it."""
    print(f"Rotating image by {degrees} degrees...")
    with Image.open(editor_dict["composite"]) as img:
        return img.convert("L").rotate(degrees, expand=True), img


def save_rotated_image(survey_date, _page, output_img):
    """Save rotated image to OUTPUT_PATH."""
    output_dir = OUTPUT_PATH
    if output_img is None:
        return "Error: No image available to save."
    try:
        os.makedirs(output_dir, exist_ok=True)
        _date = survey_date.replace("-", "") if survey_date else "undated"
        filename = f"{output_dir}/{_date}_{_page}.jpg"
        output_img.save(filename, quality=95)
        return filename
    except Exception as e:
        return f"Error: {str(e)}"


def codes2names():
    """Load band codes mapping for eBird validation."""
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
    data_date = df.loc[0, "Date"].split(" ")[0].replace("-", "")
    df['Date'] = pd.to_datetime(df.Date).dt.strftime("%m/%d/%Y")
    df['Seen'] = df.Seen.apply(lambda x: int(x) if x!="-" else 0)
    df['Heard'] = df.Heard.apply(lambda x: int(x) if x!="-" else 0)
    df["Number"] = df[["Seen", "Heard"]].sum(axis=1)
    df['Seen'] = df.Seen.apply(lambda x: f"{x} seen" if x!=0 else None)
    df['Heard'] = df.Heard.apply(lambda x: f"{x} heard" if x!=0 else None)

    # Combine Seen/Heard counts to species record comment
    _comments = df[["Seen", "Heard"]].values
    species_comments = []
    for c in _comments:
        vals = [x for x in c if isinstance(x, str)]
        species_comments.append(", ".join(vals))
    df["Species Comments"] = species_comments

    # Format timestamps
    times = pd.DataFrame(df["Start Time"].sort_values().drop_duplicates().reset_index(drop=True))
    times["Location Name"] = ROUTES["Seward West"][:len(times)]
    times[["Latitude", "Longitude", ]] = [None, None]
    times['start_time'] = pd.to_datetime(times["Start Time"], format="%H:%M").dt.strftime("%H:%M")
    times[report_cols[10:-1]] = ["WA", "US", "stationary", "1", "5", "yes", None, None]

    # Order records and survey stations by survey start time
    _df = df.merge(times, on=["Start Time"])
    _df["Start Time"] = _df.start_time.copy()
    result = band_codes.merge(_df, left_on="Code", right_on="Validated Code")
    result = result[report_cols]
    result.to_csv(f"{OUTPUT_PATH}/{data_date}_report.csv")
    return result


def setup_data(_most_likely=NBP_LIST, _aos_full=AOS_LIST):
    """Load reference species codes from CSVs."""
    most_likely = pd.read_csv(_most_likely)['Code'].tolist()
    aos_full = set(pd.read_csv(_aos_full)['Code'].tolist())
    return most_likely, aos_full


def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings."""
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
    """Find closest match in a target pool using Levenshtein distance."""
    matches = {}
    for guess in candidates:
        guess_clean = str(guess).upper().strip()
        if not guess_clean:
            continue
        if guess_clean in target_pool:
            matches[0] = [guess_clean]
            return matches
        matches[1] = []
        matches[2] = []
        for valid_code in target_pool:
            dist = levenshtein_distance(guess_clean, valid_code)
            if dist == 1:
                matches[1].append(valid_code)
                if len(matches[1]) > 10:
                    break
            elif dist == 2:
                matches[2].append(valid_code)
                if len(matches[2]) > 10:
                    break
        if len(matches[1]) > 0:
            del (matches[2])
            return matches
        elif len(matches[2]) > 0:
            del (matches[1])
            return matches
        else:
            return None


def match_code_to_tiered_lexicon(guesses, likely_codes, unlikely_codes):
    """Evaluate guesses against tiered lexicon: likely first, then unlikely."""
    likely_upper = {code.upper() for code in likely_codes}
    unlikely_upper = {code.upper() for code in unlikely_codes}
    likely_matches = search_pool(guesses, likely_upper)
    if likely_matches is not None:
        likely_dist = np.min(list(likely_matches.keys()))
        if likely_dist > 0:
            best_likely_matches = ', '.join(likely_matches[likely_dist])
            print(f"-> Best matches: {best_likely_matches} (distance: {likely_dist})")
        return likely_matches
    unlikely_matches = search_pool(guesses, unlikely_upper)
    if unlikely_matches is not None:
        unlikely_dist = np.min(list(unlikely_matches.keys()))
        best_unlikely_matches = ', '.join(unlikely_matches[unlikely_dist])
        print(f"-> Unlikely pool match: {best_unlikely_matches} (distance: {unlikely_dist})")
        return unlikely_matches
    else:
        print(f"-> No matches for {guesses} found; manual validation required")
        return None


def extract_json_from_string(raw_response):
    """Extract and parse JSON from text response."""
    try:
        text = str(raw_response).strip()
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        json_str = match.group(1) if match else text
        json_str = json_str.strip()
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            print(f"Error: No valid JSON found in text:\n{text}")
            return None
        json_clean = json_str[start_idx:end_idx]
        return json.loads(json_clean)
    except Exception as e:
        print(f"Failed to extract JSON from raw text. Error Details: {e}")
        raise


def load_ocr_page_file(page_num, date):
    """Load OCR results from a saved page file if it exists.
    Args:
        page_num: 1 or 2
        date: formatted date as YYYYMMDD
    Returns:
        pandas.DataFrame or None if file doesn't exist
    """
    filename = f"{OUTPUT_PATH}/{date}_page{page_num}.json"
    try:
        df = pd.read_json(filename)
        if df is not None and len(df) > 0:
            print(f"Loaded saved OCR results: {filename}")
            return df
    except FileNotFoundError:
        print(f"Page {page_num} file not found: {filename}")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
    return None


def set_ocr_stopped(stopped=True):
    """Set the OCR stopped flag for session state management."""
    global OCR_STOPPED
    OCR_STOPPED = stopped
    if stopped:
        print("OCR stopped by user")


def run_ocr_pipeline(model_name, date_picker, p1_img_array, p2_img_array):
    """Executes OCR via OpenAI API compatible endpoint and returns flattened table.
    Args:
        model_name: Name of OCR model to use
        date_picker: Survey date in YYYY-MM-DD format
        p1_img_array: Page 1 image (numpy array)
        p2_img_array: Page 2 image (numpy array)
    Returns:
        tuple: (header dict, filename of combined JSON)
    """
    global OCR_STOPPED

    if OCR_STOPPED:
        print("Abort signal received. Cannot resume OCR.")
        ocr_status = "Stopped"
        ocr_progress = 0
        ocr_current_page = 0
        return {"error": "OCR processing was stopped."}, "stopped_error.json"

    prompt = """
    You are an expert ornithologist transcribing field notes from a survey.
    The image is a handwritten page from a birding notebook. Pages may include a header
    with date (month/day) and a brief description of weather (temperature in degrees F, wind,
    precipitation, cloud cover). A typical header entry will look like this: "12/7, 38, light rain"
    Each page will have four data blocks with a timestamp (hours:minutes) in the upper left.
    The data blocks will be located upper left, upper right, lower left, lower right.
    Tabular format: [four letter code] | [Count 1] | [Count 2].
    A typical row of data will look like: "BEWR    -    3"  and each data block may contain
    between one and 25 rows of data. Most data blocks will contain between five and ten rows.
    Analyze the tabular content in this document and extract the data column-by-column.
    The first column will always contain four capitalized letters and will never contain numbers or special characters.
    The second and third columns are counts and will contain either integers, white space or "-". At least one of these
    columns MUST contain an integer count value. White space between characters always indicates a column break.
    You must return your response inside a valid JSON markdown block exactly like this:
    {"header": {date and weather summary},
    "data_block n": {
    "start_time": "hr:min",
    "records": [
    {"primary_code_guess": "BEWR",
    "seen": "1 or -",
    "heard": "2 or -"}]}}
    """

    output_frames = []
    headers = {0: None, 1: None}
    most_likely, aos_full = setup_data()
    _date = dt.strptime(date_picker, "%Y-%m-%d").date()
    formatted_date = dt.strftime(_date, format="%m/%d/%Y")

    # Reset progress and page state for new run
    ocr_current_page = 0
    ocr_progress = 0

    # Run the same process for page1 and page2 images
    output_frames = []
    comments = ""
    for i, img_array in enumerate([p1_img_array, p2_img_array]):
        base64_image = np_image_to_base64(img_array)

        try:
            response = client.responses.create(
                model=model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "original"}
                        ],
                    }
                ],
                include=["message"],
                temperature=0.3,
                top_logprobs=10
            )

            raw_text = response.output_text
            extracted_data = extract_json_from_string(raw_text)

            # Flatten structural JSON into flat rows for gr.DataFrame
            rows = []
            headers[i] = extracted_data.pop("header", None)
            try:
                comments = headers[i]['weather']
            except (KeyError, TypeError):
                continue
            for block_name, block_content in extracted_data.items():
                if not isinstance(block_content, dict) and block_name != 'header':
                    continue
                start_time = block_content.get("start_time", "-")
                for record in block_content.get("records", []):
                    code = record.get("primary_code_guess", "")
                    validated = match_code_to_tiered_lexicon([code], most_likely, aos_full) if code else ""
                    if isinstance(validated, dict):
                        keys = [k for k in validated.keys()]
                        keys.sort()
                        dist = keys[0]
                        valid_code = ", ".join(validated[dist])
                    else:
                        dist, valid_code = "NA", "NA"
                    rows.append({
                        "Block": block_name,
                        "Date": formatted_date,
                        "Start Time": start_time,
                        "Primary Guess": code,
                        "Seen": record.get("seen", "0"),
                        "Heard": record.get("heard", "0"),
                        "Validated Code": valid_code,
                        "Distance": dist,
                        "Submission Comments": comments
                    })

            df = pd.DataFrame(rows)
            output_file = f"{OUTPUT_PATH}/{dt.strftime(_date, format="%Y%m%d")}_page{i+1}.json"
            df.to_json(output_file)
            output_frames.append(df)
            ocr_status = f"Page {i+1} saved"
            ocr_progress = i * 50
            ocr_status = f"Processing page {i+1}, model {model_name}"

        except Exception as e:
            return pd.DataFrame([["API Error", str(e), "-", "-", "-", "-"]], columns=["Block", "Start Time", "Primary Guess", "Seen", "Heard", "Validated Code"]), {"error": str(e)}

    combined = pd.concat(output_frames).reset_index(drop=True)
    combined_filename = f"{OUTPUT_PATH}/{dt.strftime(_date, format="%Y%m%d")}_combined.json"
    combined.to_json(combined_filename)
    # Reset progress state for next run
    ocr_progress = 0
    ocr_current_page = 0
    ocr_status = "Ready"
    return headers, combined_filename


def encode_image_to_base64(image_path):
    """Encodes an image to base64 for API transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def np_image_to_base64(img_array):
    """Encodes an image to base64 for API transmission."""
    img = Image.fromarray(img_array)
    buff = BytesIO()
    img.save(buff, format="PNG")
    img_bytes = buff.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def load_multiple_files(f1, f2):
    df1 = load_test_df(f1)
    df2 = load_test_df(f2)
    try:
        return pd.concat([df1, df2])
    except ValueError as e:
        print(e)
        return None


def load_test_df(path):
    if path is None:
        return None
    return pd.read_json(path)


def df_to_csv(df, path):
    """Export DataFrame to CSV file."""
    filename = str(path).split(".")[0] + "_validated.csv"
    df.to_csv(filename)
    return filename
