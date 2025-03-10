import setuptools
from setuptools import setup

install_deps = ['numba>=0.60.0', 
                # 'numpy>=1.22.4', 
                'scipy',
                'fastremap','scikit-image',
                'mahotas>=1.4.13']

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name="ncolor",
    license="BSD",
    author="Kevin Cutler",
    author_email="kevinjohncutler@outlook.com",
    description="label matrix coloring algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevinjohncutler/ncolor",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    setup_requires=[
      'pytest-runner',
      'setuptools_scm',
    ],
    use_scm_version=True,
    install_requires = install_deps,
    tests_require=[
      'pytest'
    ],
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    )
)
