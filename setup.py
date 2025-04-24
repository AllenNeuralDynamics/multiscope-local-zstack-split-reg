from setuptools import setup, find_packages

setup(
    name="local_zstack_split_reg",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "opencv-python",
        "scanimage-tiff-reader"
    ],
    # extras_require={
    #     'scanimage': ['ScanImageTiffReader'],
    # },
    author="Jinho Kim",
    author_email="jinho.kim@alleninstitute.org",
    description="Package for processing local z-stack images with split channels from Multiscope",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
) 