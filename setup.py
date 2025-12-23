from setuptools import setup, find_packages


setup(
    name     = 'traininglib',
    packages = ['traininglib', 'traininglib.segmentation', 'traininglib.paths'],
    install_requires = [
        'torch>=2.0',
        'torchvision>=0.15'
    ],
)
