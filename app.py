import streamlit as st

st.set_page_config(
    page_title="Student Performance Support System",
    page_icon="🎓",
    layout="wide",
)

st.title("Student Performance Decision-Support System")

st.write(
    """
    This application helps teachers and school administrators identify
    students who may require and benefit from additional academic support.
    """
)

st.info(
    """
    This prediction is intended to support professional judgment from trained academic professionals.
    """
)