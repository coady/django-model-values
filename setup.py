from setuptools import setup
import model_values

setup(
    name='django-model-values',
    version=model_values.__version__,
    description='Taking the O out of ORM.',
    long_description=open('README.rst').read(),
    author='Aric Coady',
    author_email='aric.coady@gmail.com',
    url='https://bitbucket.org/coady/django-model-values',
    license='Apache Software License',
    py_modules=['model_values'],
    install_requires=['django>=1.7'],
    tests_require=['pytest-django', 'pytest-cov', 'django-dynamic-fixture'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Framework :: Django',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Database :: Database Engines/Servers',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
