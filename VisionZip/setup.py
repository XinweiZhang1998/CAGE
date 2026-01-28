from setuptools import setup, find_packages

setup(
    name="visionzip",
    version="0.1.3",
    packages=find_packages(where='visionzip'),
    author="Senqiao Yang",
    author_email="yangsenqiao.ai@gmail.com",
    description="VisionZip: Longer is Better but Not Necessary in Vision Language Models",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dvlab-research/visionzip",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License", 
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
