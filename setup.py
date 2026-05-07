from setuptools import find_packages, setup


setup(
    name="universal_speech_enhancement",
    version="0.1.0",
    description="Universal Speech Enhancement Policy Learning platform",
    packages=find_packages(where="."),
    include_package_data=True,
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "torch",
        "torchaudio",
        "transformers",
    ],
)
