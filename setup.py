from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
        name = 'atomap',
        packages = [
            'atomap',
            'atomap.tests',
            'atomap.tests.datasets',
            'atomap.external',
            'atomap.example_data',
            ],
        version = '0.1.3',
        description = 'Library for analysing atomic resolution images',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author = 'Magnus Nord',
        author_email = 'magnunor@gmail.com',
        license = 'GPL v3',
        url = 'http://atomap.org/',
        download_url = 'https://gitlab.com/atomap/atomap/repository/archive.tar?ref=0.1.3',
        keywords = [
            'STEM',
            'data analysis',
            'microscopy',
            ],
        install_requires = [
            'scipy',
            'numpy>=1.13',
            'h5py',
            'matplotlib>=2.0',
            'tqdm',
            'scikit-learn',
            'hyperspy>=1.4',
            'pillow>=5.3',
            ],
        classifiers = [
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            ],
        package_data = {
            'atomap.tests.datasets': [
                'test_ADF_cropped.hdf5',
                'test_ABF_cropped.hdf5',
                'test_atom_lattice.hdf5'],
            'atomap.example_data': [
                'example_detector_image.hspy'],
            }
)
