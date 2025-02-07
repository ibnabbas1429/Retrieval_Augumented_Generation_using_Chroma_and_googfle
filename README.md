# Creating a document question and answer Retrieval Augumented Generation using Chroma

 A RAG system has three stages:

Indexing
Retrieval
Generation
Indexing happens ahead of time, and allows you to quickly look up relevant information at query-time. When a query comes in, you retrieve relevant documents, combine them with your instructions and the user's query, and have the LLM generate a tailored answer in natural language using the supplied information. This allows you to provide information that the model hasn't seen before, such as product-specific knowledge or live weather updates.

In this notebook you will use the Gemini API to create a vector database, retrieve answers to questions from the database and generate a final answer. You will use Chroma, an open-source vector database. With Chroma, you can store embeddings alongside metadata, embed documents and queries, and search your documents.
