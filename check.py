import importlib
import sys

def check_package(name, min_version=None):
    try:
        pkg = importlib.import_module(name)
        version = getattr(pkg, "__version__", "Unknown")
        if min_version and version != "Unknown":
            # Simple lexicographical comparison (works for most semver)
            status = "✅" if version >= min_version else "⚠️ (Version low)"
            print(f"{status} {name}: {version} (Required: >= {min_version})")
        else:
            print(f"✅ {name}: {version}")
    except ImportError:
        print(f"❌ {name}: NOT INSTALLED")

print(f"Python version: {sys.version.split()[0]}\n")
print("--- Checking Requirements ---")

requirements = [
    ("numpy", "1.23"),
    ("pandas", "1.5"),
    ("scipy", None),
    ("matplotlib", None),
    ("sklearn", "1.2"),  # scikit-learn is imported as 'sklearn'
    ("jupyterlab", None),
    ("torch", None),
    ("torchmetrics", None),
]

for name, ver in requirements:
    check_package(name, ver)

print("\n--- Hardware Check ---")
try:
    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
except:
    print("Torch not available for hardware check.")