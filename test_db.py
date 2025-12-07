from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

def test_mongo_connection():
    uri = "mongodb://localhost:27017/"
    print(f"Attempting to connect to {uri}...")
    
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        print("SUCCESS: Connected to MongoDB!")
        
        db = client["idTracking_db"]
        count = db["tracked_objects"].count_documents({})
        print(f"Database 'idTracking_db' is accessible. Documents in 'tracked_objects': {count}")
        
    except ConnectionFailure:
        print("ERROR: Server not available. Is MongoDB running?")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_mongo_connection()
