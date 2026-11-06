import streamlit as st
from pydantic import ValidationError

from main import ExtractRequest, extract_from_doc_endpoint

st.title("Legal Document Extractor (Streamlit UI)")
workspace_id = st.text_input("Enter Workspace ID (workspace_id):")
doc_id = st.text_input("Enter Document ID (doc_id):")
prompt = st.text_area("Optional Prompt:", value="")

if st.button("Extract from Document"):
    if not workspace_id or not doc_id:
        st.error("Please enter both workspace_id and doc_id.")
    else:
        try:
            req = ExtractRequest(
                workspace_id=workspace_id,
                doc_id=doc_id,
                prompt=prompt if prompt else None,
            )
            result = extract_from_doc_endpoint(req)
            st.success("Extraction complete!")
            st.json(result)
        except ValidationError as exc:
            st.error(f"Input validation error: {exc}")
        except Exception as exc:
            st.error(f"Extraction failed: {exc}")
