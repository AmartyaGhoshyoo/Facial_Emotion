from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from Chroma_Embed import chroma_collection
import chromadb.utils.embedding_functions as embedding_functions

google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="AIzaSyDy66sAL6y7d2kvq6ON4UeQ0x3CVj4G4Ho")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AIzaSyDy66sAL6y7d2kvq6ON4UeQ0x3CVj4G4Ho"
)


user_query = "I am sad"
collection=chroma_collection(google_ef)
results = collection.query(
    query_texts=[user_query],
    n_results=1 
    )

print(results['documents'][0][0])
print(results['metadatas'][0][0])

messages = [
    SystemMessage(
        content=(
            "You are an expert assistant for emotion "
            "Using the following retrieved context to wrtie nicely to the user's question. "
            "Do not use external knowledge. If the answer is not in the context, clearly state that the information is unavailable. "
            f"\n\nContext:\n{results['metadatas'][0][0]}"
        )
    ),
    HumanMessage(content=user_query)
]

# --- Get response ---
response = llm.invoke(messages)
print("\n🤖 Gemini Response:\n", response.content)
