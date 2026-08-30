from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from langchain_mistralai import ChatMistralAI

from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel

from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser


class Movies(BaseModel):

    summary: str

    cast: Optional[List[str]] = None

    title: Optional[str] = None

    year: Optional[int] = None

    rating: Optional[float] = None


parser = PydanticOutputParser(pydantic_object=Movies)


model = ChatMistralAI(

    model="mistral-small-2506",

    temperature=0

)


prompt = ChatPromptTemplate.from_messages([

    (

        "system", """extract movie information from the given paragraph and provide the output in the following format:

        {format_instructions}"""

    ),

    ("human", "{paragraph}")

])


# ---------------- STREAMLIT UI ----------------

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Information Extractor")

st.write(
    "Enter a movie paragraph and extract the movie information using AI."
)

paragraph = st.text_area(
    "Enter the cinema paragraph:",
    height=300,
    placeholder="Paste your movie paragraph here..."
)


if st.button("🎯 Extract Movie Information", use_container_width=True):

    if paragraph.strip():

        with st.spinner("Extracting movie information..."):

            final_prompt = prompt.invoke({

                "paragraph": paragraph,

                "format_instructions": parser.get_format_instructions()

            })

            response = model.invoke(final_prompt)

        st.success("Movie information extracted successfully!")

        st.subheader("========== MOVIE INFORMATION ==========")

        st.write(response.content)

    else:

        st.warning("Please enter a cinema paragraph first.")