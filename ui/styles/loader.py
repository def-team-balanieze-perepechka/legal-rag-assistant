from pathlib import Path
import streamlit as st

_STYLES_DIR = Path(__file__).parent

@st.cache_resource
def _read_css(filename: str) -> str:
    return (_STYLES_DIR / filename).read_text(encoding="utf-8")

def inject_css(filename: str = "style.css") -> None:
    st.html(f"<style>{_read_css(filename)}</style>")