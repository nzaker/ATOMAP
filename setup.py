from distutils.core import setup
setup(
        name = 'atomap',
        packages = ['atomap'],
        version = '0.01',
        description = 'Library for analysing atomic resolution images',
        author = 'Magnus Nord',
        author_email = 'magnunor@gmail.com',
        url = 'https://gitlab.com/atomap/atomap',
        install_requires=[
            'scipy',
            'numpy>=1.10',
            'h5py',
            'ipython>=2.0',
            'matplotlib>=1.2',
            'hyperspy>=1.1.1',
            ],
)
