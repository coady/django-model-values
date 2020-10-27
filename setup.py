from setuptools import setup

setup(
    name='django-model-values',
    version='1.2',
    description='Taking the O out of ORM.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Aric Coady',
    author_email='aric.coady@gmail.com',
    url='https://github.com/coady/django-model-values',
    project_urls={'Documentation': 'https://coady.github.io/django-model-values'},
    license='Apache Software License',
    py_modules=['model_values'],
    install_requires=['django>=2.2'],
    python_requires='>=3.6',
    tests_require=['pytest-django', 'pytest-cov'],
    keywords='values_list pandas column-oriented data mapper pattern orm',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Framework :: Django :: 2.2',
        'Framework :: Django :: 3.0',
        'Framework :: Django :: 3.1',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Database :: Database Engines/Servers',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
