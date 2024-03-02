# Recommending Tea Blends using Retrieval augmented generation (RAG)(Product recommendation)

> An example chat
> ![image](https://github.com/vak-01/RAG-recommendation-system/assets/78302047/65f3703f-65c8-41fb-ae6d-01ac81d2c6cf)

## What is Retrieval Augmented Generation? :thinking:
Retrieval Augmented Generation is a powerful NLP technique that combines the best of retrieval-based and generative models. In simple terms, it’s like having a conversation with a knowledgeable assistant who fetches relevant information from a vast database before crafting a response.

## How does it work? :gear:
RAG operates in two steps:

- **Retrieval**: It first retrieves a set of documents from a knowledge source that are relevant to the input query.
- **Generation**: It then uses a generative model to construct a response, taking into account both the input query and the retrieved documents.
This two-step process allows RAG to generate responses that are not only contextually relevant but also rich in factual details.

Similarly, this system works in two main steps:

1. **Retrieval**: When a user asks a question, the system uses the `gpt-3.5-turbo` model to generate a set of search terms. These search terms are then used to retrieve relevant documents from the database using semantic search.
2. **Recommendation**: The retrieved documents are ranked based on their similarity to the user's input, and the top-ranked documents are returned as recommendations and are fed into the model for enabling context in the conversation.

## Why is it important? :dart:
RAG is a game-changer because it effectively bridges the gap between retrieval-based and generative models, bringing together the strengths of both. It has wide-ranging applications, from chatbots and question-answering systems to content generation and beyond.

## About this project
The system uses 
- `gpt-3.5-turbo` model to generate search terms,
- `text-embedding-ada-002` model from OpenAI to generate text embeddings.
- `cosine similarity` to calculate similarity scores.

The database of tea blends is a JSON file, and each document in the database is represented by an embedding vector.




## Getting started with the project

1. Make sure python is installed on your system.

2. Get your OpenAI developer API key from the OpenAI developer portal
   
3. Clone the repo
      ```md
      git clone https://github.com/vak-01/RAG-recommendation-system.git
      ```
4. `cd` into `RAG-Recommendation-system`

5. Create a virtual environment for the project and activate it
      ```md
      python -m venv env
      env\Scripts\activate
      ```

6. Install the requirements
    ```
    pip install -r requirements.txt
    ```

7. Create a .env file at the root of your project.

8. In the `.env` file, enter your OpenAI API key as follows - `OPENAI_API_KEY=<your openai api key goes here>`

9. Create the embeddings for the document (needed only if new data has to embedded)
      ```py
      python embed_documents.py
      ```

10. Run the chatbot
      ```py
      python retrieval_recommender.py
      ```

## Notes:
- The script uses OpenAI SDK for python (openai module) for making the requests.
- The number of API requests permitted per unit time(a minute) may vary over time, modify the script if requests are exceeding the permissible limits.


## Scope for improvement:
This project can be made into a web app for better accessibility.
