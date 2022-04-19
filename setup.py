import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ml_pipeline',
    version='0.0.1',
    author='Davor Korman',
    author_email='davor.korman@clstrlobe.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mike-huls/toolbox',
    # project_urls = {
    #     "Bug Tracker": "https://github.com/mike-huls/toolbox/issues"
    # },
    license='Clstrlobe',
    # packages=['ml_pipeline', 'test_package', 'test_package2'],
    packages=setuptools.find_packages(),
    install_requires=[
        'joblib>=1.1.0',
        'matplotlib>=3.5.1',
        # 'matplotlib-base>=3.5.1',
        'numpy>=1.20.3',
        # 'numpy-base>=1.20.3',
        'pandas>=1.4.1',
        # 'pip>=21.2.4',
        'pymssql>=2.2.3',
        # 'python>=3.9.12',
        'pyyaml>=6.0',
        'scikit-learn>=1.0.2',
        'scipy>=1.7.3',
        # 'setuptools>=61.2.0',
        'shap>=0.39.0',
        'sqlalchemy>=1.4.32',
        # 'sqlite>=3.38.2',
        'category-encoders>=2.4.0',
        'feast>=0.19.4',
        # 'feast-postgres>=0.2.5',
        'statsmodels>=0.13.2',
        'xgboost>=1.5.1',
        'mlflow>=1.22.0',
        'seaborn>=0.11.2'
    ]
)
