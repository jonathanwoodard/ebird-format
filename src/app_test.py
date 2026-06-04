# import streamlit as st
# import pandas as pd
# from PIL import Image
# import mlx_vlm
# from mlx_vlm.utils import load, generate
# import io
# import json

# # --- 1. SETTINGS & MODEL LOADING ---
# st.set_page_config(page_title="Bird Note Digitizer", layout="wide")

# @st.cache_resource
# def get_model():
#     # Loading the 26B model (4-bit) once and caching it
#     model, processor = load("mlx-community/gemma-4-26b-it-4bit")
#     return model, processor

# @st.cache_data
# def get_reference_data():
#     most_likely = pd.read_csv('most_likely.csv')['Code'].tolist()
#     aos_full = set(pd.read_csv('aos_full.csv')['Code'].tolist())
#     return most_likely, aos_full

# # --- 2. CONFIGURATION & ROUTES ---
# ROUTES = {
#     "Seward East": ["Seward East St 1", "Seward East St 2", "Seward East St 3", "Seward East St 4", 
#                     "Seward East St 5", "Seward East St 6", "Seward East St 7", "Seward East St 8"],
#     "Seward West": ["Seward West St 1", "Seward West St 2", "Seward West St 3", "Seward West St 4", 
#                     "Seward West St 5", "Seward West St 6", "Seward West St 7", "Seward West St 8"]
# }

# # --- 3. SESSION STATE INITIALIZATION ---
# if "extraction_results" not in st.session_state:
#     st.session_state.extraction_results = [] # Stores lists of records

# # --- 4. UI: SIDEBAR ---
# with st.sidebar:
#     st.title("Settings")
#     selected_route = st.selectbox("Select Survey Route", options=list(ROUTES.keys()))
#     survey_date = st.date_input("Survey Date")
#     uploaded_file = st.file_uploader("Upload Notebook Page", type=["jpg", "jpeg", "png"])
    
#     if st.button("Clear All Progress"):
#         st.session_state.extraction_results = []
#         st.rerun()

# # --- 5. IMAGE PROCESSING LOGIC ---
# def process_page(img, page_side, likely_codes):
#     model, processor = get_model()
#     width, height = img.size
#     mid = width // 2
    
#     # Split logic
#     crop_box = (0, 0, mid, height) if page_side == "Left" else (mid, 0, width, height)
#     cropped_img = img.crop(crop_box)
    
#     prompt = f"""<|user|>
# <|image|>
# Transcribe these bird notes. Format as JSON. 
# Likely codes: {", ".join(likely_codes[:50])}
# Extract 4 groups. Each group has a 'time' and a list of 'records' with 'species', 'c1', 'c2'.
# <|assistant|>"""

#     output = mlx_vlm.generate(model, processor, cropped_img, prompt, max_tokens=1500, temp=0.0)
#     # Basic cleaning to get JSON from model response
#     try:
#         return json.loads(output[output.find("{"):output.rfind("}")+1])
#     except Exception as e:
#         print(f'Failed to process inputs with error {e}')
#         return None

# # --- 6. MAIN APP FLOW ---
# st.title(f"Survey Digitizer: {selected_route}")

# if uploaded_file:
#     img = Image.open(uploaded_file)
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.image(img, caption="Original Scan", use_container_width=True)
    
#     with col2:
#         st.subheader("Extraction Control")
#         side = st.radio("Which side is this?", ["Left", "Right"])
        
#         if st.button("Run OCR"):
#             with st.spinner(f"Gemma 4-26B analyzing {side} page..."):
#                 likely, _ = get_reference_data()
#                 data = process_page(img, side, likely)
#                 if data:
#                     # Flatten the data for the table
#                     flat_records = []
#                     for group in data.get("groups", []):
#                         for rec in group.get("records", []):
#                             flat_records.append({
#                                 "Time": group.get("time"),
#                                 "Species": rec.get("species"),
#                                 "Count_1": rec.get("c1"),
#                                 "Count_2": rec.get("c2")
#                             })
#                     st.session_state.temp_df = pd.DataFrame(flat_records)
#                 else:
#                     st.error("Failed to parse JSON. Try again.")

#     # --- 7. EDITABLE DATA TABLE ---
#     if "temp_df" in st.session_state:
#         st.divider()
#         st.subheader("Edit & Validate Data")
#         st.info("Check species codes against your reference list before saving.")
        
#         edited_df = st.data_editor(
#             st.session_state.temp_df,
#             num_rows="dynamic",
#             use_container_width=True,
#             key="editor"
#         )
        
#         if st.button("Save Page Data to Survey"):
#             st.session_state.extraction_results.append(edited_df)
#             st.success(f"Added 4 points to survey. Total points captured: {len(st.session_state.extraction_results) * 4}")
#             del st.session_state.temp_df

# # --- 8. EBIRD EXPORT ---
# if len(st.session_state.extraction_results) >= 2:
#     st.divider()
#     st.header("Finalize eBird Upload")
    
#     if st.button("Generate eBird CSV"):
#         # Combine all data
#         all_data = pd.concat(st.session_state.extraction_results)
        
#         # eBird Record Format requires specific columns
#         # Map your points to the Route locations
#         route_points = ROUTES[selected_route]
        
#         ebird_rows = []
#         # Logical loop: each group of records under one 'Time' is a checklist
#         unique_times = all_data['Time'].unique()
        
#         for i, time_val in enumerate(unique_times):
#             point_name = route_points[i] if i < len(route_points) else f"Point {i+1}"
#             subset = all_data[all_data['Time'] == time_val]
            
#             for _, row in subset.iterrows():
#                 total_count = (row['Count_1'] or 0) + (row['Count_2'] or 0)
#                 if total_count == 0: 
#                     total_count = "X"
                
#                 ebird_rows.append({
#                     "Common Name": "", # Leave empty if using Species Code
#                     "Genus": "",
#                     "Species": row['Species'],
#                     "Number": total_count,
#                     "Species Comments": "",
#                     "Location Name": point_name,
#                     "Latitude": "",
#                     "Longitude": "",
#                     "Date": survey_date.strftime("%m/%d/%Y"),
#                     "Start Time": time_val,
#                     "State/Province": "",
#                     "Country Code": "",
#                     "Protocol": "Stationary",
#                     "Number of Observers": 1,
#                     "Duration (Min)": 5,
#                     "All observations reported?": "Y",
#                     "Effort Distance Miles": "",
#                     "Effort area acres": ""
#                 })
        
#         ebird_df = pd.DataFrame(ebird_rows)
#         csv = ebird_df.to_csv(index=False).encode('utf-8')
        
#         st.download_button(
#             label="Download eBird CSV",
#             data=csv,
#             file_name=f"ebird_{selected_route}_{survey_date}.csv",
#             mime="text/csv"
#         )









import io
import os
import json
import tempfile
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime as dt
from huggingface_hub import scan_cache_dir
from streamlit_cropperjs import st_cropperjs

# Import your custom local pipeline modules
from ocr_preprocess import rotate_image, find_deskew_angle
import ocr_extract


# ---------------------------------------------------------
# CACHE SCANNER FOR HUGGINGFACE
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def get_local_hf_models():
    """Scans the local HF cache directory for downloaded models."""
    try:
        cache_info = scan_cache_dir()
        # Filter for models containing 'mlx' or 'vlm' in their names
        models = [repo.repo_id for repo in cache_info.repos if repo.repo_type == "model"]
        mlx_vlm_models = [m for m in models if "mlx" in m.lower() or "vlm" in m.lower()]
        return mlx_vlm_models if mlx_vlm_models else models
    except Exception:
        default_cache = os.path.expanduser("~/.cache/huggingface/hub")
        st.sidebar.warning(f"Using default path fallback: {default_cache}")
        return []

# ---------------------------------------------------------
# STREAMLIT UI CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Modular Local MLX-VLM OCR")
st.title("⚙️ Modular Local OCR Pipeline Engine")

# --- Sidebar Configuration ---
st.sidebar.header("Pipeline Settings")

survey_date = st.sidebar.date_input("Select the date of the survey", dt.today())
_date = dt.strftime(survey_date, "%Y%m%d")
available_models = get_local_hf_models()
if not available_models:
    selected_model = st.sidebar.text_input("No models cached. Enter Model ID manually:", "mlx-community/nanoLLaVA")
else:
    selected_model = st.sidebar.selectbox("Select Cached MLX Model Path", available_models)

_most_likely = st.sidebar.file_uploader(
    "Select likely species list", 
    type=["csv"], 
    accept_multiple_files=False
)
most_likely = pd.read_csv(_most_likely)['Code'].tolist()

_aos_full = st.sidebar.file_uploader(
    "Select full AOS species list", 
    type=["csv"], 
    accept_multiple_files=False
)
aos_full = set(pd.read_csv(_aos_full)['Code'].tolist())


# --- Layout Columns ---
col_upload, col_table = st.columns([1, 1])
try:

    with col_upload:
        st.subheader("1. File Selection")
        uploaded_files = st.file_uploader(
            "Select raw image file(s)", 
            type=["png", "jpg", "jpeg"], 
            accept_multiple_files=True
        )
        
        # Track paths to pass to your scripts
        uploaded_paths = []
        
        if uploaded_files:
            preview_cols = st.columns(len(uploaded_files))
            
            # Create a persistent directory for this session run to hold the files
            # Streamlit runs linearly, so we clean/re-create these temporary paths on upload change
            temp_dir = tempfile.gettempdir()
            
            for idx, file in enumerate(uploaded_files):
                # Save uploaded buffer bytes into an actual local system file path string
                temp_file_path = os.path.join(temp_dir, f"st_upload_{_date}_p{idx + 1}.jpg")
                with open(temp_file_path, "wb") as f:
                    f.write(file.getvalue())
                    
                    uploaded_paths.append(temp_file_path)
                    
                    # Render preview directly from the system path
                    st.subheader("2. Crop Image to Text")

                    img = rotate_image(temp_file_path)
                    preview_cols[idx].image(img, caption=f"Original: {file.name}", width="stretch")
                    image_bytes = io.BytesIO()
                    img.save(image_bytes, format="JPEG")
                    pic = image_bytes.getvalue()
                    cropped = st_cropperjs(pic=pic, btn_text="Crop Image")
                    if cropped is not None:
                        cropped_img = Image.open(io.BytesIO(cropped)).convert("L")
                        st.image(cropped_img, caption=f"Cropped Image {temp_file_path}", width="stretch")
                        cropped_img.save(temp_file_path, "JPEG", quality=95)
except Exception as e:
    st.error(f"Image Processing Interrupted: {str(e)}")
st.success("Preprocessing sequence completed!")

# Trigger Pipeline Execution
can_execute = len(uploaded_paths) == 2 and selected_model
if st.button("Run Local OCR Extraction", disabled=not can_execute, type="primary"):
    st.info(f"Step 2: Feeding paths into {selected_model} extraction engine...")
    with st.spinner("Processing pipeline step by step..."):
        try:
            ocr_results = []
            # Passing the list of 2 file paths and the selected model ID string
            for idx, path in enumerate(uploaded_paths):
                structured_output = ocr_extract.extract_text(path, selected_model, most_likely, aos_full)
            
                # Expecting your extract script to return a python dictionary or list of objects
                if isinstance(structured_output, str):
                    structured_output = json.loads(structured_output)
                ocr_results.append(structured_output)
            # Save the successfully returned tabular records into state
            st.session_state["ocr_records"] = ocr_results
            st.success("Extraction pipeline complete!")
            
        except Exception as e:
            st.error(f"Pipeline Interrupted: {str(e)}")

with col_table:
    frames = []
    st.subheader("2. Interactive Table & Export JSON")
    # Fallback default mock layout until pipeline creates real records
    if "ocr_records" not in st.session_state:
        st.session_state["ocr_records"] = [
            {"Field": "Example Invoice Row", "Value": "0.00", "Confidence": "Run Pipeline to update"}
        ]
    # Cast python object layout to DataFrame for editing interface
    df = pd.DataFrame(st.session_state["ocr_records"])
    
    # Provide spreadsheet editing capability
    edited_df = st.data_editor(
        df, 
        width="stretch", 
        num_rows="dynamic",
        key="main_table_editor"
    )
    # else:
    #     for record in st.session_state["ocr_records"]:
    #         # Fallback default mock layout until pipeline creates real records
    #         # Cast python object layout to DataFrame for editing interface
    #         df = pd.DataFrame(record)
    #         frames.append(df)
    #     data = pd.concat(frames)
    # Provide spreadsheet editing capability
    # edited_df = st.data_editor(
    #     data, 
    #     width="stretch", 
    #     num_rows="dynamic",
    #     key="main_table_editor"
    # )
    
    # Instantly output modified grid elements into dynamic tabular json orientation
    final_json = edited_df.to_dict(orient="records")
    
    st.write("---")
    st.caption("Live Interactive JSON Code Block:")
    st.json(final_json)
    
    # Save/Download to Local Machine
    st.download_button(
        label="Download Edited JSON Payload",
        data=json.dumps(final_json, indent=4),
        file_name="pipeline_output.json",
        mime="application/json"
    )
