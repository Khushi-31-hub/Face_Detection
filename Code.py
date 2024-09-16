This is your main Python script. Make sure to include comments and docstrings explaining the functionality.

  import cv2

def initialize_face_classifier():
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_classifier.empty():
        raise RuntimeError("Error loading face cascade. The XML file might be missing or corrupt.")
    return face_classifier

def detect_faces(frame, face_classifier):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def main():
    face_classifier = initialize_face_classifier()
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open video capture.")
        return
    
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame from video capture.")
            break
        
        faces = detect_faces(frame, face_classifier)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Save the frame to a file
        filename = f"frame_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        frame_count += 1

        # Optionally, print the filename or other information
        print(f"Saved frame to {filename}")
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
