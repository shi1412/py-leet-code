from setuptools import setup

description='A library for analytic processing: including runs ad hoc queries against database to generate reports and ML results'
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError, RuntimeError):
    long_description = description

setup(
    name='pyleetcode',
    packages=['pyleetcode'],
    description=description,
    long_description=long_description,
    author='Michael Shi',
    url='https://github.com/shi1412/py-leet-code',
    keywords=['leet-code', 'data-structure', 'algorithms'],
    install_requires=['pytest==6.2.1'],
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.7',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'License :: OSI Approved :: MIT License',
    ],
    entry_points={
        'console_scripts': [
           
        ],
    },
)
