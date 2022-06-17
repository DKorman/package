import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='test package',
    version='0.0.1',
    author='Davor Korman',
    author_email='',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='none',
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas>=1.4.1'
    ]
)
