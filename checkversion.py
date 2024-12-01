from importlib.metadata import version

# List of packages to check
pkgs = [
    "matplotlib",
    "numpy",
    "tiktoken",
    "torch",
    "tensorflow",  # For OpenAI's pretrained weights
    "pandas"       # Dataset loading
]

# Print the version of each package
def check_package_versions():
    for p in pkgs:
        try:
            print(f"{p} version: {version(p)}")
        except Exception as e:
            print(f"Error checking version for {p}: {e}")

if __name__ == "__main__":
    check_package_versions()
