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
    
class TeaRecommender:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.max_message_length = 5000 
        self.messages = []
        with open('system_message_retrieval', 'r') as file:
            self.system_message = file.read()
        self.messages.append({"role": "system", "content": self.system_message})

    # a method which check if the all content within the messages list is more than 30000 words if so we drop the earlier messages but not the system message
    def check_messages_length(self):
        messages_length = 0
        for message in self.messages:
            # Splitting the content into words and adding the count to messages_Length
            messages_length += len(message['content'].split(' '))
        
        if messages_length > self.max_message_length:
            # drop the earlier messages but not the system message. Lets do this one by one until the Length is Less than 300. Keep removing index 1 as index 0 is the system message
            while messages_length > self.max_message_length:
                messages_length -= len(self.messages[1]['content']. split())
                self.messages.pop(1)

    def recommend(self,role, input):
        self.messages.append({"role": role, "content": input})    
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                stream=True,
                messages=self.messages
            )
        except Exception as e:
            print(f"Error: {e}")

        responses = ''
        user_response_printing = True
        printTeaRecommendation = True
        first_chunk = True

        # print('---------------------')

        for chunk in response:
            if chunk.choices[0].delta.content:
                text_chunk = chunk.choices[0].delta.content
                # print(text_chunk)
                responses += str(text_chunk)

                try:
                    if 'your response to the user": "' in responses:
                        user_response_printing = True
                        if printTeaRecommendation:
                            print("\n")
                            print(colored("Tea bot: ", "green"))
                            printTeaRecommendation = False

                        if first_chunk:
                            if first_chunk:
                                text_chunk = text_chunk.lstrip(' "')
                                first_chunk = False

                        if '"' or '}' in text_chunk:
                            # count the number of times the character '"' appears in the text_chunk
                            # if the character '"' appears odd number of times, remove the Last character
                            if text_chunk.count('"') % 2 != 0:
                                text_chunk = text_chunk.rstrip('"\n')
                            text_chunk = text_chunk.rstrip('}')
                        print(colored(text_chunk, "yellow"), end="", flush=True)
                except json.JSONDecodeError:
                    pass

                if not user_response_printing:
                    print(text_chunk, end="", flush=True) # Print chunks as they arrive

        print("\n-------------------------------------")
        self.messages.append({"role": "assistant", "content": responses})
        # drop the earlier messages but not the system message
        self.check_messages_length()
        return json.loads(responses)


if __name__ == "__main__":
    print(colored("\nHello, I am Tea Bot. I can help you with tea recommendations. Ask me anything about tea.\n", "red"), end="", flush=True)
    recommender = TeaRecommender()
    while True:
        print(colored("User : ", "blue"), end="", flush=True)
        user_input = input()
        recommender_response = recommender.recommend("user", user_input)

        # check to see if we have retrieval
        if "search term for retrieval" in recommender_response:
            if PRINT_SEARCH_RESULTS:
                print(f"Retrieval: {recommender_response['search term for retrieval']}")

            similarity_search = SimilaritySearch('modified_json_file.json')

            # Lets Limit the number of search terms to 3 just in case GPT returns more than 3 search terms
            results = similarity_search.search(recommender_response['search term for retrieval'][:3])
            if PRINT_SEARCH_RESULTS:
                print("SIMILARITY SEARCH RESULTS:")
                print(results)
            retrieved_tea_blends = []
            # return top 3 results from each query and append it to retrieved_tea_blends without embeddings and similarities
            
            for result in results: # each result is a list of similarity results for each query
            # we want to retriev 3 results from each query without the embeddings and similarities
                for record in result.head(3).to_dict('records'):
                    # remove the "embeddings" key
                    record.pop('embeddings', None)
                    # remove the similarities key
                    record.pop('similarities', None)
                    retrieved_tea_blends.append(record)
            
            if PRINT_SEARCH_RESULTS:
                print("RETRIEVED TEA BLENDS:")
                print(retrieved_tea_blends)
            
            # append the retrieved condles to the messages
            recommender.messages.append({"role": "assistant", "content": f"these are the tea blends matching your search terms:\n\n {retrieved_tea_blends}"})

            recommender.recommend("assistant", "keep these recommendations in mind during your conversation with the user and now respond to the user")