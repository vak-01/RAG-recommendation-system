import os
from openai import OpenAI
from termcolor import colored
import numpy as np
import pandas as pd
import json
from typing import List
from dotenv import load_dotenv

load_dotenv()

PRINT_SEARCH_RESULTS = False              # Set to True to print search results

class SimilaritySearch:
    def __init__(self, json_file_path: str):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        self.df = pd.DataFrame(self.data)

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_embedding(self, text, model="text-embedding-ada-002"):
        return self.client.embeddings.create(input = [text], model=model).data[0].embedding

    def search(self, user_query: List[str], top_n=3, to_print=True):
        results = []
        for query in user_query:
            embedding = self.get_embedding(query, model="text-embedding-ada-002")
            self.df["similarities"] = self.df.embeddings.apply(lambda x: self.cosine_similarity(x[1], embedding[1]))
            res = self.df.sort_values("similarities", ascending=False).head(top_n)
            if to_print and PRINT_SEARCH_RESULTS:
                print(f"Search query: {query}")
                print(res[["name", "similarities"]])
            results.append(res)
        return results