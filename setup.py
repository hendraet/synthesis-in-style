from pathlib import Path

import setuptools

requirements_file = Path(__file__).resolve().parent / 'stylegan_code_finder' / 'requirements.txt'
requirements = [requirement.strip() for requirement in requirements_file.open().readlines() if not requirement.startswith('git')]
requirements.append('pytorch-training @ git+https://github.com/Bartzi/pytorch-training.git')

setuptools.setup(
    name="synthesis-in-style",
    version="1.0",
    author="",  # TODO:
    author_email="",  # TODO
    description="",  # TODO
    packages=setuptools.find_packages(),
    install_requires=requirements,
    dependency_links=['https://download.pytorch.org/whl/cu113'],
    classifiers=[  # TODO
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3.8',
)
