import cv2                   #In parole semplici: è il "cervello" che permette ai computer di "vedere" e capire cosa c'è in un'immagine o in un video

class VideoInputFacade:      #Inizializza la sorgente video
    def __init__(self, source_path): #parametro  video_source: Percorso del file video (es. "assets/video.mp4") oppure 0 per la webcam
                                
        self.video_source = source_path

    # Se source_path è un numero (es. 0), lo converte in int per la webcam
        if str(source_path).isdigit():                                          #questo controllo serve a capire se l'input è una stringa o un numero , se è una stringa e quindi un mercorso di un video lo apre altrimenti lo converte in un numero e in base al numero esegue derminati comportamenti per esempio se metto 0 si riferisce alla webcam di defaultdel pc , se metto 1 alla webcam esterna collegata tramite usb eccusb ecc 
            source_path = int(source_path)
            
        self.capture = cv2.VideoCapture(source_path)
        
        if not self.capture.isOpened():
            raise ValueError(f"Errore: Impossibile aprire il video o la webcam: {source_path}")

    def get_frame(self):
        """
        Restituisce il prossimo frame del video.
        :return: Il frame (immagine) se disponibile, altrimenti None (fine video).
        """
        ret, frame = self.capture.read()

        cv2.imshow('Frame', frame)
        
        if not ret:
            return None
        return frame

    def get_video_info(self):
        """
        Restituisce larghezza, altezza e FPS. Utile per salvare il video output dopo.
        """
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        return width, height, fps

    def release(self):
        """
        Chiude correttamente la risorsa video.
        """
        self.capture.release()
        cv2.destroyAllWindows()
    

    