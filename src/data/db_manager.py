from pymongo import MongoClient
from datetime import datetime

class DBManager:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="idTracking_db", collection_name="tracked_objects"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        print(f"Connected to MongoDB: {db_name}.{collection_name}")

    def update_object_plate(self, obj_id, plate_text):
        """
        Updates the document for the given object ID with the detected license plate.
        If the document doesn't exist, it creates one.
        """
        # Convert numpy types to native Python types for MongoDB
        if hasattr(obj_id, 'item'):
            obj_id = obj_id.item()

        filter_query = {"track_id": obj_id}
        update_query = {
            "$set": {
                "plate": plate_text,
                "last_updated": datetime.now()
            },
            "$setOnInsert": {
                "created_at": datetime.now()
            }
        }
        self.collection.update_one(filter_query, update_query, upsert=True)
        print(f"DB: Updated object {obj_id} with plate '{plate_text}'")
        print(f"DB: Updated object {obj_id} with plate '{plate_text}'")

    def save_detection(self, obj_data):
        """
        Saves a raw detection record (optional, if we want a history of all detections).
        """
        obj_data["timestamp"] = datetime.now()
        self.collection.insert_one(obj_data)
