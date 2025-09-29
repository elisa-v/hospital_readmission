import argparse

def run_prediction(input_path: str, output_path: str) -> None:
    print("🔮 Running predictions...")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    # TODO: load model from models/, read input CSV, write predictions to output_path

def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="results/predictions.csv", help="Path to save predictions CSV")
    args = parser.parse_args()
    run_prediction(args.input, args.output)

if __name__ == "__main__":
    main()
