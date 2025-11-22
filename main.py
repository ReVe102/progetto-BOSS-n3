import cv2
from src.input_ouput.video_facade import VideoInputFacade
from src.processing.detector import ObjectDetector # <--- NUOVO IMPORT

def main():
    video_path = "assets/video4.mp4" # O 0 per webcam
    
    try:
        # 1. Inizializziamo i moduli
        video_loader = VideoInputFacade(video_path)
        detector = ObjectDetector() # <--- Carichiamo YOLO
        
        print("Avvio sistema...")
        
        while True:
            frame = video_loader.get_frame()
            
            if frame is None:
                break 

            # 2. Eseguiamo il Tracking
            detections = detector.detect_and_track(frame)
            
            # 3. Disegniamo i risultati (solo per debug visivo)
            for obj in detections:
                id = obj['id']
                x1, y1, x2, y2 = obj['bbox']
                
                # Disegna rettangolo verde
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Scrivi l'ID sopra
                cv2.putText(frame, f"ID: {id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("SafeDrive - Test Tracking", frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
        video_loader.release()
        
    except Exception as e:
        print(f"Errore: {e}")

if __name__ == "__main__":
    main()