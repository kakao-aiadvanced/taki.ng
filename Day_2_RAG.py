import os
import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "api-token-key"

llm = ChatOpenAI(model="gpt-4o-mini")

# Load, chunk and index the contents of the blog.
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 임베딩 모델 지정 (text-embedding-3-small 사용)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 유사도 기반 검색
    search_kwargs={"k": 6}  # top 6개 결과 반환
)

parser = JsonOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def my_template(instruction):
    return instruction + """\n
Here is the document:
{context}

Here is the user's question:
{question}

{format_instructions}
"""


prompt = PromptTemplate(
    template= my_template("""
    You are a helpful assistant that checks if the user's question is relevant to the given document.
your answer is only relevance=yes or relevance=no json object."""
                         ),
    input_variables=["context", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt_yes = PromptTemplate(
    template=my_template("answer the question."),
    input_variables=["context", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt_no = PromptTemplate(
    template=my_template("answer 'No'."),
    input_variables=["context", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


template_hallucination = """
check hallucination. your answer is only hallucination=yes or hallucination=no json object.

Here is the document:
{context}

Here is the user's question:
{answer}

{format_instructions}
"""

prompt_hallucination = PromptTemplate(
    template=template_hallucination,
    input_variables=["context", "answer"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
rag_chain = prompt_yes | llm | StrOutputParser()
rag_chain_no = prompt_no | llm | StrOutputParser()
rag_chain_hallucination = prompt_hallucination | llm | StrOutputParser()



## test
query = "agent memory"

relevant_docs = retriever.invoke(query)

relevance_results = []
for doc in relevant_docs:
    result = chain.invoke({
        "context": doc.page_content,
        "question": query
    })
    relevance_results.append(result["relevance"])

print(relevance_results)

if 'yes' in relevance_results:
    answer = rag_chain.invoke({
            "context": retriever | format_docs,
            "question": query
        })
    hallucination_result = rag_chain_hallucination.invoke({
            "context": retriever | format_docs,
            "answer": answer
        })
    print(hallucination_result)

else:
    answer = rag_chain_no.invoke({
            "context": retriever | format_docs,
            "question": query
        })
    print(answer)


