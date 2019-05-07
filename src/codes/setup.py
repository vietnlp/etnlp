from setuptools import setup, find_packages
from etnlp_api import __version__


with open("../../README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='ETNLP',
    version=__version__,
    # packages=['api', 'utils', 'embeddings', 'visualizer'],
    packages=find_packages(),
    py_modules=['etnlp_api'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/vietnlp/etnlp',
    license='MIT',
    author='vietnlp',
    author_email='sonvx.coltech@gmail.com',
    description='ETNLP: Embedding Toolkit for NLP Tasks'
)
# from setuptools import setup, find_packages
# import sys
#
# with open('requirements.txt') as f:
#     reqs = f.read()
# setup(
#     name='ETNLP',
#     version='0.1.0',
#     description='ETNLP: Embedding Toolkit for NLP Tasks',
#     python_requires='>=3.5',
#     packages=find_packages(exclude=('data')),
#     install_requires=reqs.strip().split('\n'),
# )
