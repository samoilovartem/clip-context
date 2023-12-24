from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import NoTranscriptFound

load_dotenv()

embeddings = OpenAIEmbeddings()


def create_vector_db_from_youtube_url(youtube_url: str) -> FAISS:
    """
    Creates a vector database from a YouTube video url
    :param youtube_url: The YouTube url
    :return: FAISS(The vector database)
    """
    try:
        loader = YoutubeLoader.from_youtube_url(youtube_url=youtube_url)
        transcript = loader.load()
    except NoTranscriptFound:
        raise ValueError(
            'No English transcript could be retrieved for this video. Please ensure the video has an '
            'English transcript available.'
        )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)

    return db


def get_response_from_query(query: str, db: FAISS, k: int = 4) -> str:
    """
    Gets a response from a query
    :param k: Number of documents to return
    :param query: The query
    :param db: The vector database
    :return: The response
    """
    docs = db.similarity_search(query, k=k)
    docs_page_content = ' '.join([doc.page_content for doc in docs])

    llm = OpenAI(model_name='text-davinci-003')

    prompt = PromptTemplate(
        input_variables=['query', 'docs'],
        template="""
        As a knowledgeable assistant, your role is to provide answers to questions regarding YouTube videos,
        utilizing the content of the video's transcript.

        Please address the question: {query}
        Refer to the transcript of this video for information: {docs}

        Your response should be based solely on the factual details contained within the transcript.

        In instances where the information in the transcript is insufficient to form a complete answer,
        please respond with "Insufficient information to determine an answer."

        Strive to deliver answers that are not only accurate but also rich in detail and explanatory depth.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(query=query, docs=docs_page_content)
    response.replace('\n', '')

    return response
