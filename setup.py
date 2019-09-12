from setuptools import setup


setup(name='pytope',
      version='0.0.3',
      description='Polytope operations --- minimal functionality',
      url='https://github.com/heirung/pytope',
      author='Tor Aksel N. Heirung',
      author_email='github@torheirung.com',
      license='MIT',
      packages=['pytope'],
      install_requires=[
        'numpy',
        'scipy>=1.3.0',  # 1.3.0 for the reflective-simplex option in linprog()
        'pycddlib',
        'matplotlib'
      ],
      zip_safe=False)
