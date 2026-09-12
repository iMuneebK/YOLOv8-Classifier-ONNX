from ultralytics import YOLO
import cv2
from typing import Union

class YOLOInference:
    def __init__(self, model_path: str, task: str = 'detect'):
        self.model = YOLO(model_path)
        self.task = task
        
    def predict_image(self, image_path: str):
        """Run inference on a single image."""
        results = self.model(image_path)
        return results[0]

    def predict_video(self, video_path: str, output_path: str):
        """Run inference on a video and save output."""
        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
        cap.release()
        out.release()
        
    def predict_webcam(self):
        """Run inference on webcam feed."""
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model(frame)
            cv2.imshow("YOLOv8 Inference", results[0].plot())
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
