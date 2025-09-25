from setuptools import setup, find_packages
setup(
  name = 'cfg-pytorch',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '0.7.1',
  license='MIT',
  description = 'Classifier Free Guidance - Pytorch',
  long_description_content_type = 'text/markdown',
  install_requires=[
    'beartype',
    'einops>=0.8',
    'environs',
    'ftfy',
    'open-clip-torch>=2.8.0',
    'torch>=2.0',
    'transformers[torch]'
  ],
  
)