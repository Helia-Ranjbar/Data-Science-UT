import subprocess

pipeline_steps = [
    "load_data.py",
    "preprocess.py",
    "feature_engineering.py",
    "data_to_db.py",
    "train_model.py",
    "make_predictions.py"
]

def run_step(script_name):
    print(f"\n Running: {script_name}")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error in {script_name}:\n{result.stderr}")
        raise RuntimeError(f"{script_name} failed")
    
    print(f"Completed: {script_name}")
    print(result.stdout)

def main():
    print("Starting full ML pipeline...\n")
    try:
        for script in pipeline_steps:
            run_step(script)
        print("\n All pipeline steps completed successfully.")
    except Exception as e:
        print(f"\n Pipeline halted: {e}")

if __name__ == "__main__":
    main()
