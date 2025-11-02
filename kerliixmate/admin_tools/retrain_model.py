import subprocess

def retrain_model():
    print("Starting fine-tuning pipeline...")
    # Call the training script
    subprocess.run(["python", "../kerliixmate_training.py"])

if __name__ == "__main__":
    retrain_model()
