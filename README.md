# Recommending Tea Blends using Retrieval augmented generation (RAG)(Product recommendation)


The system works in two main steps:

1. **Retrieval**: When a user asks a question, the system uses a model to generate a set of search terms. These search terms are then used to retrieve relevant documents from a database of tea blends.

2. **Recommendation**: The retrieved documents are ranked based on their similarity to the user's input, and the top-ranked documents are returned as recommendations.

The system uses 
- `gpt-3.5-turbo` model to generate search terms,
- `text-embedding-ada-002` model from OpenAI to generate text embeddings.
- `cosine similarity` to calculate similarity scores.

The database of tea blends is stored in a JSON file, and each document in the database is represented by an embedding vector.


> An example chat
> ![image](https://github.com/vak-01/RAG-recommendation-system/assets/78302047/65f3703f-65c8-41fb-ae6d-01ac81d2c6cf)
