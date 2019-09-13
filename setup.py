from setuptools import setup

with open('README.md', 'r') as fh:
  long_description = fh.read()

setup(name='pytope',
      version='0.0.4',
      description='Polytope operations --- limited functionality',
      long_description=long_description,
      long_description_content_type='text/markdown',
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
      python_requires='>=3.6',
      zip_safe=False)
