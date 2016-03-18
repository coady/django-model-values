About django-model-values
=========================
.. image:: https://img.shields.io/pypi/v/django-model-values.svg
   :target: https://pypi.python.org/pypi/django-model-values/
.. image:: https://img.shields.io/pypi/pyversions/django-model-values.svg
.. image:: https://img.shields.io/pypi/status/django-model-values.svg
.. image:: https://img.shields.io/travis/coady/django-model-values.svg
   :target: https://travis-ci.org/coady/django-model-values
.. image:: https://img.shields.io/codecov/c/github/coady/django-model-values.svg
   :target: https://codecov.io/github/coady/django-model-values

Provides `Django`_ model utilities for encouraging direct data access instead of unnecessary object overhead.
Implemented through compatible method and operator extensions to ``QuerySets`` and ``Managers``.

The goal is to provide elegant syntactic support for best practices in using Django's ORM.
Specifically avoiding the inefficiencies and race conditions associated with always using objects.

Usage
=========================
*Do you want readability, ...*

.. code-block:: python

   book = Book.objects.get(pk=pk)
   book.rating = 5.0
   book.save()

*efficiency, correctness, ...*

.. code-block:: python

   Book.objects.filter(pk=pk).update(rating=5.0)

*Choose all 3*

.. code-block:: python

   Book.objects[pk]['rating'] = 5.0

Instantiate the custom ``Manager`` in your models.
See `documentation`_ for more examples.

Installation
=========================
Standard installation from pypi or local download. ::

   $ pip install django-model-values
   $ python setup.py install

Dependencies
=========================
   * Django 1.8+
   * Python 2.7, 3.3+

Tests
=========================
100% branch coverage. ::

   $ py.test [--cov]

Changes
=========================
0.3

   * Lookup methods and operators
   * F expressions and aggregation methods
   * Database functions
   * Conditional expressions for updates and annotations
   * Bulk updates and change detection

0.2

   * Change detection
   * Groupby functionality
   * Named tuples

.. _django: https://docs.djangoproject.com
.. _documentation: http://pythonhosted.org/django-model-values/
