from ultralytics import YOLO
import sys

def export_model_to_onnx(model_path: str):
    """Exports a trained YOLOv8 model to ONNX format."""
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print("Exporting to ONNX format...")
    # Export the model
    path = model.export(format='onnx', imgsz=640, optimize=True)
    print(f"Export successful! ONNX model saved at: {path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        export_model_to_onnx(sys.argv[1])
    else:
        print("Usage: python onnx_export.py <path_to_pt_model>")
