from langchain.embeddings import OpenAIEmbeddings
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

def get_embedding(text: str) -> list:
    embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    # Get the embeddings for the text
    res = embeddings_model.embed_documents([text])[0]
    # Return the embeddings
    return res

# Function to embed a chunk of text
def embed_chunk(i, text):
    #.Get the embeddings for the text
    embeddings = get_embedding(text)
    print(" <--- embedded the document", i+1, " ---> ")
    return i, embeddings

def embed_documents_in_json_file(input_file_path: str, output_file_path: str) -> None:
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)["tea_blends"]

    print("--- Embedding the documents --- ")

    for i, item in enumerate(data):
        if "embeddings" not in item.keys():
            json_object = json.dumps(item)
            embeddings = embed_chunk(i, json_object)
            data[i]["embeddings"] = embeddings

            # Sleep for 20 seconds (1/3 of a minute) to respect the API's rate limit
            time.sleep(20)

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, separators=(',', ':'))

    print(" ... done saving the embeddings ... ")
    print()

if __name__== "__main__":
    input_file_path = "mock_tea_data.json"
    output_file_path = "mock_tea_data_embeddings.json"
    embed_documents_in_json_file(input_file_path, output_file_path)