import cv2
import os

def images_to_video():
    # 1. PERCORSI (Modifica se necessario)
    image_folder = 'assets/0001'
    video_name = 'assets/kitti_0004.mp4'
    fps = 10  

    # 2. Controllo cartella
    if not os.path.exists(image_folder):
        print(f"Errore: Cartella non trovata: {image_folder}")
        return

    # 3. Prendi tutte le immagini e ORDINALEE
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # Fondamentale: ordiniamo per numero (00001, 00002...)
    images.sort() 

    if not images:
        print("Nessuna immagine trovata!")
        return

    # 4. Leggi la prima immagine per capire le dimensioni
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    size = (width, height)

    print(f"Trovate {len(images)} immagini. Creazione video in corso...")

    # 5. Configura il VideoWriter
    # 'mp4v' è il codec standard per .mp4 su Windows/OpenCV
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    # 6. Scrivi i frame uno per uno
    for i, image_name in enumerate(images):
        img_path = os.path.join(image_folder, image_name)
        img = cv2.imread(img_path)
        out.write(img)
        
        if i % 50 == 0:
            print(f"Processato frame {i}/{len(images)}")

    out.release()
    print(f"✅ Fatto! Video salvato in: {video_name}")

if __name__ == "__main__":
    images_to_video()