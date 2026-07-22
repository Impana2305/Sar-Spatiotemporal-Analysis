import os
import sys
import subprocess

# Force UTF-8 stdout encoding for Windows compatibility
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

notebooks = [
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_preprocessing.ipynb",
    "notebooks/03_autoencoder.ipynb",
    "notebooks/04_training_classifier.ipynb",
    "notebooks/05_prediction_pipeline.ipynb",
    "notebooks/06_Model_comparision.ipynb",
    "notebooks/07_Transformer.ipynb"
]

print("=== Executing all 7 notebooks sequentially to populate cell outputs ===")

for nb_path in notebooks:
    print(f"\n--- Executing {nb_path} ---")
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
        nb_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[SUCCESS] Successfully executed and saved outputs for {nb_path}")
    else:
        print(f"[ERROR] Error executing {nb_path}:")
        print(res.stderr)
        print("Standard Output:", res.stdout)
        sys.exit(1)

print("\n=== All 7 notebooks executed successfully with all cell outputs populated! ===")
