from setuptools import find_packages, setup

REQUIREMENTS = ["huggingface-hub", "requests", "rich", "safetensors", "torch"]

setup(
    name="candle-tensor-tools",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "candle-convert = candle_tensor_tools.model_conversion.cli:main"
        ]
    },
    # install_requires=REQUIREMENTS,
)
