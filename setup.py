from setuptools import setup, find_packages

setup(
    name="vannameivision",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scikit-image==0.19.3",
        "tensorflow==2.12.0",
        "tensorflow-addons==0.21.0"
    ],
    package_data={
        'vannameivision': ['../model/DenseNet121-Triplet-ImageNet.h5', '../image/*']
    },
    url="https://github.com/kobachii-d/VannameiVision",
    author="Kobchai Duangrattanalert",
    author_email="kduangrattanalert@gmail.com",
    description="VannameiVision uses advanced state-of-the-art technologies to detect susceptible shrimp larvae with 92% accuracy. By integrating probabilistic deep learning, transfer learning, and deep metric learning, this tool has been demonstrated to perform effectively and consistently across various backgrounds, enhancing larvae detection."
)
