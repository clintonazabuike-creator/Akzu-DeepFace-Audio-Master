import os
import sys
import argparse
import cv2
import numpy as np
import onnxruntime as ort

from color_matcher import match_colors_lab

def parse_args():
    parser = argparse.ArgumentParser(description="High-Performance CUDA Video Inference Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Path to source face image.")
    parser.add_argument("--input", type=str, default="0", help="Input video path or webcam index (default: 0).")
    parser.add_argument("--output", type=str, default="", help="Optional output path to save processed video.")
    parser.add_argument("--execution-provider", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference hardware backend.")
    parser.add_argument("--execution-threads", type=int, default=8, help="Number of intra/inter op threads.")
    return parser.parse_args()

def initialize_onnx_session(model_path: str, provider: str, threads: int):
    """Initializes ONNX Runtime session with strict CUDA or CPU providers."""
    providers = []
    if provider == "cuda":
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append(("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 4 * 1024 * 1024 * 1024, # 4GB VRAM ceiling cap
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            }))
            print("[INFO] CUDA Execution Provider successfully bound.")
        else:
            print("[WARNING] CUDA requested but not available. Falling back to CPUExecutionProvider.")
    
    providers.append("CPUExecutionProvider")
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = threads
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model weights not found at: {model_path}")
        sys.exit(1)
        
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)

def main():
    args = parse_args()
    
    print("==========================================================")
    print("       DEEPFAKE VIDEO PIPELINE - RTX CUDA RUNTIME         ")
    print("==========================================================")
    print(f"Source Image       : {args.source}")
    print(f"Input Stream       : {args.input}")
    print(f"Execution Provider : {args.execution_provider.upper()}")
    print(f"Execution Threads  : {args.execution_threads}")
    print("==========================================================\n")
    
    # Load source face image
    if not os.path.exists(args.source):
        print(f"[ERROR] Source face image missing: {args.source}")
        sys.exit(1)
        
    source_img = cv2.imread(args.source)
    
    # Handle video capture source (Int or String path)
    cap_source = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(cap_source)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream: {args.input}")
        sys.exit(1)
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    print(f"[INFO] Stream Resolution: {width}x{height} @ {fps:.2f} FPS")
    print("[INFO] Press 'Q' in the display window to terminate stream.\n")
    
    # Main processing loop
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video stream or feed disconnected.")
                break
                
            # --- Placeholder for core face-swap inference injection ---
            # In live integration, face detector, embedder, and swap model run here.
            # Applying LAB color harmonization test pass on frame:
            processed_frame = frame.copy() # Swap step will modify this region
            
            # Example call to our color matcher module if region is swapped:
            # processed_frame = match_colors_lab(swapped_face_region, original_face_region, mask)
            
            # Display real-time output
            cv2.imshow("Akzu-DeepFace-Audio-Master [CUDA Stream]", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] User termination signal received.")
                break
                
    KeyboardInterrupt
    except Exception as e:
        print(f"[ERROR] Runtime exception encountered: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Video pipeline safely closed.")

if __name__ == "__main__":
    main()
