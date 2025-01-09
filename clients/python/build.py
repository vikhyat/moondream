import toml
import sys
import shutil
from pathlib import Path
import copy

BASE_CONFIG = {
    "tool": {
        "poetry": {
            "name": "moondream",
            "version": "0.0.2",
            "description": "Python client library for moondream",
            "authors": ["vik <vik@moondream.ai>"],
            "readme": "README.md",
            "packages": [{"include": "moondream", "from": "."}],
            "dependencies": {
                "python": "^3.10",
                "pillow": "^10.4.0",
                "numpy": "^2.1.2",
            },
            "extras": {
                "cpu": [
                    "onnxruntime-1.19.2",
                    "tokenizers-0.20.1",
                ],
                "gpu": [
                    "torch-2.5.0",
                    "safetensors-0.4.2",
                    "einops-0.7.0",
                    "pyvips-binary-8.16.0",
                    "pyvips-2.2.1",
                    "tokenizers-0.20.1",
                ],
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

    # Add warning header to all Python files
    warning_header = """# WARNING: This is an auto-generated file. Do not edit directly.
# Any changes made to this file will be overwritten.
"""
    for py_file in dst_dir.rglob("*.py"):
        content = py_file.read_text()
        py_file.write_text(warning_header + content)


def build(variant=None):
    config = copy.deepcopy(BASE_CONFIG)
    if variant is None:
        pass
    elif variant == "gpu":
        torch_src = Path("../../moondream/torch")
        # Copy torch implementation
        torch_target = Path("moondream/torch")
        copy_torch_implementation(torch_src, torch_target)

        # Add GPU dependencies to main dependencies
        gpu_extras = config["tool"]["poetry"]["extras"]["gpu"]
        for dep in gpu_extras:
            # Split on last occurrence of hyphen to handle package names with hyphens
            name, version = dep.rsplit("-", 1)
            config["tool"]["poetry"]["dependencies"][name] = f"^{version}"
        print(f"Building configuration for {variant} variant...")
    elif variant == "cpu":
        # Add CPU dependencies to main dependencies
        cpu_extras = config["tool"]["poetry"]["extras"]["cpu"]
        for dep in cpu_extras:
            # Split on last occurrence of hyphen to handle package names with hyphens
            name, version = dep.rsplit("-", 1)
            config["tool"]["poetry"]["dependencies"][name] = f"^{version}"
        print(f"Building configuration for {variant} variant...")
    else:
        print(f"Unknown variant: {variant}")
        print("Usage: python build.py [cpu|gpu]")
        sys.exit(1)

    # Write the configuration
    toml_content = toml.dumps(config)
    print("Generated pyproject.toml content:")
    print(toml_content)
    Path("pyproject.toml").write_text(toml_content)

    return "moondream"


if __name__ == "__main__":
    variant = None
    if len(sys.argv) == 2:
        variant = sys.argv[1]

    package_name = build(variant)
    print(f"Successfully created {package_name} configuration")
