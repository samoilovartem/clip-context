import textwrap

import streamlit as st

from langchain_helper import create_vector_db_from_youtube_url, get_response_from_query

st.title('ClipContext')

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.text_input(label='Input YouTube video URL:', max_chars=50)
        query = st.text_area(
            label='Input your question about the video:', max_chars=200, key='query'
        )
        submit_button = st.form_submit_button(label='Submit')

if submit_button and youtube_url and query:
    try:
        db = create_vector_db_from_youtube_url(youtube_url)
        response = get_response_from_query(query, db)
        st.subheader('Answer: ')
        st.text(textwrap.fill(response, width=85))
    except ValueError as e:
        st.error(str(e))
