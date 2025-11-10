def chroma_collection(google_ef):
    import chromadb
    client = chromadb.PersistentClient(path="Chroma_VectorStore/chroma_db")
    collection_name = "Facial_Emotion_collection"
    collection = client.get_or_create_collection(name=collection_name, embedding_function=google_ef)
    return collection


if __name__=='__main__':
    emotion_reviews = {
    'Anger': "This image captures the intensity of anger, with furrowed brows and a tense expression.",
    'Disgust': "The subject in this image shows clear signs of disgust, perhaps reacting to something unpleasant.",
    'Fear': "Fear is palpable in this image, conveyed through wide eyes and a sense of apprehension.",
    'Happiness': "A joyful moment is frozen in this image, with a bright smile radiating happiness.",
    'Sadness': "This image evokes a sense of sadness, with a downcast expression and a hint of sorrow.",
    'Surprise': "The suddenness of surprise is evident in this image, with raised eyebrows and an open mouth.",
    'Neutral': "This image displays a neutral expression, showing no strong emotions.",
    'Contempt': "A subtle but clear look of contempt is captured in this image, with a slight smirk or raised lip."
}

    import chromadb.utils.embedding_functions as embedding_functions
    import uuid # for generating unique IDs
    google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="YOUR_API_KEY")

    documents = []
    metadatas = []
    ids = []

    for emotion, review in emotion_reviews.items():
        print(f"{emotion}: {review}")
        data = emotion  
        metadata = {"description": review}
        documents.append(data)
        metadatas.append(metadata)
        ids.append(str(uuid.uuid4()))
        
        
    collection=chroma_collection(google_ef)
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ Added {len(documents)} questions to ChromaDB collection Facial_Emotion_collection")
    user_query = "Sad"  

    results = collection.query(
        query_texts=[user_query],
        n_results=1  
    )

    print(results['documents'][0][0])
    print(results['metadatas'][0][0])
    top_doc = results['documents'][0][0]
    top_metadata = results['metadatas'][0][0]
    print("\n🔍 Top Similar Question:", top_doc)
    print("💬 Answer:", top_metadata['answer'])
    print("👨‍⚕️ Expert ID:", top_metadata['expert_id'])
    
