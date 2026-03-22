import argparse
import sys
import os
import torch
import litert_torch

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TFLite using litert-torch")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the .pt model")
    parser.add_argument("--output-path", type=str, default="model.tflite", help="Path to save the .tflite model")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    print(f"Loading PyTorch model from {args.model_path}...")
    ts_model = torch.jit.load(args.model_path)
    ts_model.eval()

    # The drone environment observation space has 6 values
    dummy_input = (torch.randn(1, 6),)

    print("Converting model to TFLite using litert-torch and TS2EPConverter...")
    from torch._export.converter import TS2EPConverter
    converter = TS2EPConverter(ts_model, dummy_input)
    exported_program = converter.convert()
    
    # Pass the GraphModule representation to litert_torch.convert
    edge_model = litert_torch.convert(exported_program.module(), dummy_input)
    
    edge_model.export(args.output_path)
    print(f"Model successfully converted and saved to {args.output_path}")

if __name__ == "__main__":
    main()
