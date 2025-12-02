from astrapy import DataAPIClient
import os

# Connect to Astra DB
def get_db():
    client = DataAPIClient()
    return client.get_database(
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        keyspace=os.getenv("ASTRA_DB_NAMESPACE")
    )

    

# Vector search (same idea as your Typescript implementation)
def vector_search(query_embedding, collection_name=os.getenv("ASTRA_DB_COLLECTION")):
    db = get_db()
    collection = db.get_collection(collection_name)

    results = collection.find(
    {},
    sort={"$vector": query_embedding},
    limit=10
    )

    docs = []
    for item in results:
        docs.append(item.get("text", str(item))) 
    return docs