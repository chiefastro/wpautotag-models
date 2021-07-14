import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wp_auto_taxonomy_models",
    version="0.0.3",
    author="Jared Rand",
    author_email="chiefastro@gmail.com",
    description="Modeling code for WP Auto Taxonomy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chiefastro/wp-auto-taxonomy-models",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas', 'numpy', 'fuzzywuzzy', 'scipy', 'scikit-learn'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
