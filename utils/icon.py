import streamlit as st




@st.cache_data
def show_icon():
    """Shows a gallery button."""

    st.markdown("""
    <style>
    .gallery-button {
        padding: 0.5rem 1rem;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<button onclick="show_gallery()" class="gallery-button">Gallery</button>', unsafe_allow_html=True)

    st.markdown('<script>function show_gallery() {window.location.href = "#gallery";}</script>', unsafe_allow_html=True)
