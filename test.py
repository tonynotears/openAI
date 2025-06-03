import yaml
import chromadb
import openai

# Set OpenAI API key
openai.api_key = ""

# Load the YAML file
with open("d:\\RAG\\preferences_schema.yaml", "r") as file:
    data = yaml.safe_load(file)

# Extract examples
examples = data.get("examples", [])

# Initialize ChromaDB client
client = chromadb.Client()

# Create a collection
collection = client.create_collection(name="preferences")

# Insert examples into ChromaDB
for example in examples:
    # Convert lists to comma-separated strings for metadata
    likes_str = ", ".join(example["likes"])
    dislikes_str = ", ".join(example["dislikes"])
    
    document = {
        "name": example["name"],
        "likes": likes_str,
        "dislikes": dislikes_str
    }
    
    collection.add(
        documents=[str(document)],
        metadatas=[{
            "name": example["name"],
            "likes": likes_str,
            "dislikes": dislikes_str
        }],
        ids=[example["name"]]  # Use the name as the unique ID
    )

# Query ChromaDB for context
def query_chromadb(query):
    results = collection.query(
        query_texts=[query],
        n_results=5  # Adjust the number of results as needed
    )
    return results["documents"]

# Use OpenAI with context from ChromaDB
def ask_openai_with_rag(query):
    context = query_chromadb(query)
    prompt = f"Based on the following context, answer the query:\n{context}\nQuery: {query}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that only uses the provided context as truth."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Example usage
query = "What does Tony like?"
response = ask_openai_with_rag(query)
print(response)

print("Data successfully stored in ChromaDB!")
