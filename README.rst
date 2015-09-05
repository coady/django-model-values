About django-model-values
=========================
`Django`_ model utilities for encouraging direct data access instead of unnecessary object overhead.
Implemented through compatible method and operator extensions QuerySets and Managers.

The goal is to provide elegant syntatic support for best practices in using Django's ORM.
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

See `documentation`_ for more examples.

Installation
=========================
Standard installation from pypi or local download. ::

   $ pip install django-model-values
   $ python setup.py install

Dependencies
=========================
   * Django
   * Python 2.7, 3.2+

Tests
=========================
100% branch coverage.  Tested against Django 1.8 and Python 2.7, 3.4. ::

  $ py.test

.. _django: https://docs.djangoproject.com
.. _documentation: http://pythonhosted.org/django-model-values/
