
import streamlit as st
from pydantic import ValidationError
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import ExtractRequest, extract_from_doc_endpoint

st.title("Legal Document Extractor (Streamlit UI)")
doc_id = st.text_input("Enter Document ID (doc_id):")
prompt = st.text_area("Optional Prompt:", value="")

if st.button("Extract from Document"):
    if not doc_id:
        st.error("Please enter a doc_id.")
    else:
        try:
            req = ExtractRequest(doc_id=doc_id, prompt=prompt if prompt else None)
            result = extract_from_doc_endpoint(req)
            st.success("Extraction complete!")
            st.json(result)
        except ValidationError as ve:
            st.error(f"Input validation error: {ve}")
        except Exception as e:
            st.error(f"Extraction failed: {e}")
