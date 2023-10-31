from setuptools import setup


setup(
    name             =  'traininglib',
    packages         = ['traininglib'],
    install_requires = [
        'torch>=2.0',
        'torchvision>=0.15'
    ],
)
