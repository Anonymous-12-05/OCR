import streamlit as st
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
import re

# Load the tokenizer and model with adjusted dtype
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained(
    'ucaslcl/GOT-OCR2_0',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map='cuda',
    torch_dtype=torch.float32,  # Use float32 as dtype
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id
)

# Move model to evaluation mode
model = model.eval().cuda()

# Function to perform OCR on the uploaded image
def perform_ocr(image):
    # Convert the image to a format suitable for the model
    image_file = image
    # Perform OCR using the model
    with torch.cuda.amp.autocast(dtype=torch.float32):
        ocr_result = model.chat(tokenizer, image_file, ocr_type='ocr')
    return ocr_result

# Function to search and highlight text
def search_and_highlight(text, keyword):
    # Use regex to highlight the keyword in the text
    highlighted_text = re.sub(f'({keyword})', r'<mark>\1</mark>', text, flags=re.IGNORECASE)
    return highlighted_text

# Streamlit App UI
st.title("OCR and Text Search Application")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform OCR on the uploaded image
    with st.spinner("Performing OCR..."):
        extracted_text = perform_ocr(uploaded_image)
    
    st.subheader("Extracted Text")
    st.write(extracted_text)

    # Keyword search
    keyword = st.text_input("Enter keyword to search in the extracted text:")
    
    if keyword:
        # Highlight the keyword in the extracted text
        highlighted_text = search_and_highlight(extracted_text, keyword)
        st.subheader("Search Results")
        st.markdown(highlighted_text, unsafe_allow_html=True)
