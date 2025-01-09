import toml
import sys
import shutil
from pathlib import Path

BASE_CONFIG = {
    "tool": {
        "poetry": {
            "version": "0.0.2",
            "description": "Python client library for moondream",
            "authors": ["vik <vik@moondream.ai>"],
            "readme": "README.md",
            "packages": [{"include": "moondream", "from": "."}],
            "dependencies": {
                "python": "^3.10",
                "pillow": "^10.4.0",
                "numpy": "^2.1.2",
                "tokenizers": "^0.20.1",
                "torch": "^2.2.0",
                "safetensors": "^0.4.2",
                "einops": "^0.7.0",
                "pyvips": "^2.2.1",
            },
            "scripts": {"moondream": "moondream.cli:main"},
        },
        "pyright": {
            "venvPath": ".",
            "venv": ".venv",
            "reportMissingParameterType": False,
        },
    },
    "build-system": {
        "requires": ["poetry-core"],
        "build-backend": "poetry.core.masonry.api",
    },
}

def copy_torch_implementation(src_dir: Path, dst_dir: Path):
    """Copy torch implementation files to destination"""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    # Create __init__.py if it doesn't exist
    init_file = dst_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()

def build(variant):
    config = BASE_CONFIG.copy()
    src_dir = Path("moondream")
    torch_src = Path("../../moondream/torch")

    if variant == "gpu":
        package_name = "moondream-gpu"
        config["tool"]["poetry"]["name"] = package_name
        config["tool"]["poetry"]["dependencies"]["onnxruntime-gpu"] = "^1.20.0"

        # Create package directory
        target_dir = Path(package_name)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(exist_ok=True)

        # Copy all python files
        for py_file in src_dir.glob("*.py"):
            shutil.copy2(py_file, target_dir)

        # Copy torch implementation
        torch_target = target_dir / "torch"
        copy_torch_implementation(torch_src, torch_target)

    elif variant == "cpu":
        package_name = "moondream"
        config["tool"]["poetry"]["name"] = package_name
        config["tool"]["poetry"]["dependencies"]["onnxruntime"] = "^1.19.2"
        
        # Copy torch implementation
        torch_target = src_dir / "torch"
        copy_torch_implementation(torch_src, torch_target)

    else:
        print(f"Unknown variant: {variant}")
        print("Usage: python build.py [cpu|gpu]")
        sys.exit(1)

    import copy
    config = copy.deepcopy(config)

    # Write the configuration
    toml_content = toml.dumps(config)
    print("Generated pyproject.toml content:")
    print(toml_content)
    Path("pyproject.toml").write_text(toml_content)

    print(f"Built configuration for {variant} variant")
    return package_name

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build.py [cpu|gpu]")
        sys.exit(1)

    package_name = build(sys.argv[1])
    print(f"Successfully created {package_name} configuration")
