import cv2
import numpy as np
import easyocr
import threading
import queue
from collections import Counter
from src.data.db_manager import DBManager

class PlateRecognizer:
    def __init__(self):
        self.ocr_available = False
        self.plate_history = {} # {obj_id: [list of detected plates]}
        self.processing_queue = queue.Queue()
        
        try:
            print("Initializing EasyOCR...")
            # gpu=False per evitare errori se non c'è una GPU NVIDIA
            self.reader = easyocr.Reader(['en'], gpu=False) 
            self.db_manager = DBManager()
            self.ocr_available = True
            print("EasyOCR and DBManager initialized successfully.")
            
            # Start background worker thread
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            print("OCR Worker thread started.")
            
        except Exception as e:
            print(f"Error initializing OCR or DB: {e}")

    def add_to_queue(self, frame, obj_id, bbox):
        """
        Adds a task to the OCR queue. Non-blocking.
        """
        if not self.ocr_available:
            return

        x1, y1, x2, y2 = bbox
        h, w, _ = frame.shape
        
        # Boundary checks
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Size check (fast fail)
        if (x2 - x1) < 40 or (y2 - y1) < 10:
            return

        # Crop and COPY the image so main thread can continue safely
        vehicle_crop = frame[y1:y2, x1:x2].copy()
        
        # Put in queue
        self.processing_queue.put((vehicle_crop, obj_id))

    def _worker(self):
        """
        Background thread that processes images from the queue.
        """
        while True:
            try:
                # Get task from queue
                vehicle_crop, obj_id = self.processing_queue.get()
                
                # Perform OCR (Heavy operation)
                plate_text = self._recognize_from_crop(vehicle_crop)
                
                if plate_text:
                    self._update_history_and_db(obj_id, plate_text)
                
                self.processing_queue.task_done()
            except Exception as e:
                print(f"Error in OCR worker: {e}")

    def _update_history_and_db(self, obj_id, plate_text):
        """
        Updates history and saves to DB if we are confident.
        """
        if obj_id not in self.plate_history:
            self.plate_history[obj_id] = []
        
        self.plate_history[obj_id].append(plate_text)
        
        # Keep only last 10 readings
        if len(self.plate_history[obj_id]) > 10:
            self.plate_history[obj_id].pop(0)
            
        # Voting system
        counts = Counter(self.plate_history[obj_id])
        most_common, count = counts.most_common(1)[0]
        
        # If we have seen this plate at least 2 times (CONFIDENCE >= 2)
        # Reduced from 3 to 2 to make it easier to confirm plates
        if count >= 3:
            print(f"CONFIRMED PLATE for ID {obj_id}: {most_common} (Confidence: {count}/{len(self.plate_history[obj_id])})")
            try:
                self.db_manager.update_object_plate(obj_id, most_common)
            except Exception as e:
                print(f"DB ERROR: Could not save plate for ID {obj_id}: {e}")

    def _recognize_from_crop(self, vehicle_crop):
        """
        Internal method to run OCR on a pre-cropped image.
        """
        # Preprocessing
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

        try:
            results = self.reader.readtext(gray, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            if not results:
                return None

            results.sort(key=lambda x: x[2], reverse=True)

            for (bbox_ocr, text, prob) in results:
                text_clean = ''.join(c for c in text if c.isalnum()).upper()
                
                if self.is_valid_plate(text_clean) and prob > 0.25:
                    print(f"DEBUG: OCR saw '{text_clean}' (prob={prob:.2f})")
                    return text_clean
        except Exception as e:
            print(f"OCR Error: {e}")
            
        return None

    # Legacy method wrapper if needed, but we should use add_to_queue
    def recognize_and_save(self, frame, obj_id, bbox):
        self.add_to_queue(frame, obj_id, bbox)
        return None # Returns None because it's async now

    def recognize(self, frame, bbox):
        # Deprecated for main loop use, but kept for compatibility
        return None

    def is_valid_plate(self, text):
        if len(text) < 5 or len(text) > 8:
            return False
        return True
