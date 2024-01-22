
import os
import glob

if __name__ == "__main__":
    test_scripts = glob.glob("test/*/test_*.py")
    
    for script in test_scripts:
        os.system(f"python {script}")
