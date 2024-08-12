from setuptools import setup


setup(
    name='NeutroCon',
    version='0.0.5',
    packages=['NeutroCon'],
    url='https://github.com/kyutekrap/NeutroCon',
    install_requires=[
        'Link @ git+https://github.com/kyutekrap/Link@main',
        'numpy'
    ]
)
