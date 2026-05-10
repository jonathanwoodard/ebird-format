import streamlit as st
import pandas as pd
from PIL import Image
import mlx_vlm
from mlx_vlm.utils import load, generate
import io
import json

# --- 1. SETTINGS & MODEL LOADING ---
st.set_page_config(page_title="Bird Note Digitizer", layout="wide")

@st.cache_resource
def get_model():
    # Loading the 26B model (4-bit) once and caching it
    model, processor = load("mlx-community/gemma-4-26b-it-4bit")
    return model, processor

@st.cache_data
def get_reference_data():
    most_likely = pd.read_csv('most_likely.csv')['Code'].tolist()
    aos_full = set(pd.read_csv('aos_full.csv')['Code'].tolist())
    return most_likely, aos_full

# --- 2. CONFIGURATION & ROUTES ---
ROUTES = {
    "Seward East": ["Seward East St 1", "Seward East St 2", "Seward East St 3", "Seward East St 4", 
                    "Seward East St 5", "Seward East St 6", "Seward East St 7", "Seward East St 8"],
    "Seward West": ["Seward West St 1", "Seward West St 2", "Seward West St 3", "Seward West St 4", 
                    "Seward West St 5", "Seward West St 6", "Seward West St 7", "Seward West St 8"]
}

# --- 3. SESSION STATE INITIALIZATION ---
if "extraction_results" not in st.session_state:
    st.session_state.extraction_results = [] # Stores lists of records

# --- 4. UI: SIDEBAR ---
with st.sidebar:
    st.title("Settings")
    selected_route = st.selectbox("Select Survey Route", options=list(ROUTES.keys()))
    survey_date = st.date_input("Survey Date")
    uploaded_file = st.file_uploader("Upload Notebook Page", type=["jpg", "jpeg", "png"])
    
    if st.button("Clear All Progress"):
        st.session_state.extraction_results = []
        st.rerun()

# --- 5. IMAGE PROCESSING LOGIC ---
def process_page(img, page_side, likely_codes):
    model, processor = get_model()
    width, height = img.size
    mid = width // 2
    
    # Split logic
    crop_box = (0, 0, mid, height) if page_side == "Left" else (mid, 0, width, height)
    cropped_img = img.crop(crop_box)
    
    prompt = f"""<|user|>
<|image|>
Transcribe these bird notes. Format as JSON. 
Likely codes: {", ".join(likely_codes[:50])}
Extract 4 groups. Each group has a 'time' and a list of 'records' with 'species', 'c1', 'c2'.
<|assistant|>"""

    output = mlx_vlm.generate(model, processor, cropped_img, prompt, max_tokens=1500, temp=0.0)
    # Basic cleaning to get JSON from model response
    try:
        return json.loads(output[output.find("{"):output.rfind("}")+1])
    except Exception as e:
        print(f'Failed to process inputs with error {e}')
        return None

# --- 6. MAIN APP FLOW ---
st.title(f"Survey Digitizer: {selected_route}")

if uploaded_file:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Original Scan", use_container_width=True)
    
    with col2:
        st.subheader("Extraction Control")
        side = st.radio("Which side is this?", ["Left", "Right"])
        
        if st.button("Run OCR"):
            with st.spinner(f"Gemma 4-26B analyzing {side} page..."):
                likely, _ = get_reference_data()
                data = process_page(img, side, likely)
                if data:
                    # Flatten the data for the table
                    flat_records = []
                    for group in data.get("groups", []):
                        for rec in group.get("records", []):
                            flat_records.append({
                                "Time": group.get("time"),
                                "Species": rec.get("species"),
                                "Count_1": rec.get("c1"),
                                "Count_2": rec.get("c2")
                            })
                    st.session_state.temp_df = pd.DataFrame(flat_records)
                else:
                    st.error("Failed to parse JSON. Try again.")

    # --- 7. EDITABLE DATA TABLE ---
    if "temp_df" in st.session_state:
        st.divider()
        st.subheader("Edit & Validate Data")
        st.info("Check species codes against your reference list before saving.")
        
        edited_df = st.data_editor(
            st.session_state.temp_df,
            num_rows="dynamic",
            use_container_width=True,
            key="editor"
        )
        
        if st.button("Save Page Data to Survey"):
            st.session_state.extraction_results.append(edited_df)
            st.success(f"Added 4 points to survey. Total points captured: {len(st.session_state.extraction_results) * 4}")
            del st.session_state.temp_df

# --- 8. EBIRD EXPORT ---
if len(st.session_state.extraction_results) >= 2:
    st.divider()
    st.header("Finalize eBird Upload")
    
    if st.button("Generate eBird CSV"):
        # Combine all data
        all_data = pd.concat(st.session_state.extraction_results)
        
        # eBird Record Format requires specific columns
        # Map your points to the Route locations
        route_points = ROUTES[selected_route]
        
        ebird_rows = []
        # Logical loop: each group of records under one 'Time' is a checklist
        unique_times = all_data['Time'].unique()
        
        for i, time_val in enumerate(unique_times):
            point_name = route_points[i] if i < len(route_points) else f"Point {i+1}"
            subset = all_data[all_data['Time'] == time_val]
            
            for _, row in subset.iterrows():
                total_count = (row['Count_1'] or 0) + (row['Count_2'] or 0)
                if total_count == 0: 
                    total_count = "X"
                
                ebird_rows.append({
                    "Common Name": "", # Leave empty if using Species Code
                    "Genus": "",
                    "Species": row['Species'],
                    "Number": total_count,
                    "Species Comments": "",
                    "Location Name": point_name,
                    "Latitude": "",
                    "Longitude": "",
                    "Date": survey_date.strftime("%m/%d/%Y"),
                    "Start Time": time_val,
                    "State/Province": "",
                    "Country Code": "",
                    "Protocol": "Stationary",
                    "Number of Observers": 1,
                    "Duration (Min)": 5,
                    "All observations reported?": "Y",
                    "Effort Distance Miles": "",
                    "Effort area acres": ""
                })
        
        ebird_df = pd.DataFrame(ebird_rows)
        csv = ebird_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download eBird CSV",
            data=csv,
            file_name=f"ebird_{selected_route}_{survey_date}.csv",
            mime="text/csv"
        )