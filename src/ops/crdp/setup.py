from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("crdp", ["crdp.pyx"], extra_compile_args=["-O3"]),
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="crdp",
    version="0.0.3",
    author="Ran Bi",
    author_email="biran0079@gmail.com",
    description="A fast Ramer-Douglas-Peucker algorithm implementation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/biran0079/crdp",
    keywords="rdp ramer douglas peucker line simplification cython",
    license="MIT",

    ext_modules=cythonize(extensions,compiler_directives={"language_level": 3}),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
