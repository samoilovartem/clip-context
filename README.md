# ClipContext

ClipContext is a very simple Streamlit application that leverages the power of language models to answer questions based on YouTube video transcripts. Users can input a YouTube URL and ask questions related to the video's content. ClipContext processes the video transcript and provides detailed, informative answers.

## Features

- Accepts YouTube video URLs and user queries through a Streamlit interface.
- Creates a searchable vector database from YouTube video transcripts.
- Utilizes OpenAI language models to generate verbose and detailed answers.
- Handles videos with English transcripts, providing user-friendly error messages when transcripts are unavailable.

## Installation

To set up your local environment to run ClipContext, follow these steps:

```bash
# Clone the repository
git clone https://github.com/samoilovartem/clip-context

# Install Poetry, if it's not already installed
pip3 install poetry

# Install the dependencies using Poetry
poetry install

# Run the Streamlit application
streamlit run main.py
