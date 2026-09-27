import argparse
from src.data_preprocessing import preprocess_pipeline
from src.train import run_training
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="End-to-End Fake and True News Detection Pipeline")
    parser.add_argument(
        "--preprocess", 
        action="store_true", 
        help="Run only the data preprocessing step"
    )
    parser.add_argument(
        "--train", 
        action="store_true", 
        help="Run only the model training step"
    )
    parser.add_argument(
        "--evaluate", 
        action="store_true", 
        help="Run only the model evaluation step"
    )
    
    args = parser.parse_args()
    
    # If no flags are set, run the entire pipeline
    run_all = not (args.preprocess or args.train or args.evaluate)
    
    if run_all or args.preprocess:
        print("\n" + "="*50)
        print("STAGE 1: Running Preprocessing...")
        print("="*50)
        preprocess_pipeline()
        
    if run_all or args.train:
        print("\n" + "="*50)
        print("STAGE 2: Running Model Training...")
        print("="*50)
        run_training()
        
    if run_all or args.evaluate:
        print("\n" + "="*50)
        print("STAGE 3: Running Model Evaluation...")
        print("="*50)
        evaluate_model()
        
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()
