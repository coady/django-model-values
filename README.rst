.. image:: https://img.shields.io/pypi/v/django-model-values.svg
   :target: https://pypi.org/project/django-model-values/
.. image:: https://img.shields.io/pypi/pyversions/django-model-values.svg
.. image:: https://img.shields.io/pypi/status/django-model-values.svg
.. image:: https://img.shields.io/travis/coady/django-model-values.svg
   :target: https://travis-ci.org/coady/django-model-values
.. image:: https://img.shields.io/codecov/c/github/coady/django-model-values.svg
   :target: https://codecov.io/github/coady/django-model-values
.. image:: https://readthedocs.org/projects/django-model-values/badge
   :target: `documentation`_

`Django`_ model utilities for encouraging direct data access instead of unnecessary object overhead.
Implemented through compatible method and operator extensions to ``QuerySets`` and ``Managers``.

The goal is to provide elegant syntactic support for best practices in using Django's ORM.
Specifically avoiding the inefficiencies and race conditions associated with always using objects.

Usage
=========================
Typical model usage is verbose, inefficient, and incorrect.

.. code-block:: python

   book = Book.objects.get(pk=pk)
   book.rating = 5.0
   book.save()

The correct method is generally supported, but arguably less readable.

.. code-block:: python

   Book.objects.filter(pk=pk).update(rating=5.0)

``model_values`` encourages the better approach with operator support.

.. code-block:: python

   Book.objects[pk]['rating'] = 5.0

Similarly for queries:

.. code-block:: python

   (book.rating for book in books)
   books.values_list('rating', flat=True)
   books['rating']

Column-oriented syntax is common in panel data layers, and the greater expressiveness cascades.
``QuerySets`` also support aggregation and conditionals.

.. code-block:: python

   books.values_list('rating', flat=True).filter(rating__gt=0)
   books['rating'] > 0

   books.aggregate(models.Avg('rating'))['rating__avg']
   books['rating'].mean()

``Managers`` provide a variety of efficient primary key based utilities.
To enable, instantiate the ``Manager`` in your models.
As with any custom ``Manager``, it doesn't have to be named ``objects``,
but it is designed to be a 100% compatible replacement.

.. code-block:: python

   from model_values import Manager

   class Book(models.Model):
      ...
      objects = Manager()

``F`` expressions are also enhanced, and can be used directly without model changes.

.. code-block:: python

   from model_values import F

   .filter(rating__gt=0, last_modified__range=(start, end))
   .filter(F.rating > 0, F.last_modified.range(start, end))

Read the `documentation`_.

Installation
=========================
::

   $ pip install django-model-values

Dependencies
=========================
* django >=1.11

Tests
=========================
100% branch coverage. ::

   $ pytest [--cov]

Changes
=========================
dev

* Transform functions
* Named tuples
* Window functions
* Distance lookups
* Django 2.1 functions
* ``EnumField``
* Annotated ``items``

0.5

* ``F`` expressions operators ``any`` and ``all``
* Spatial lookups and functions
* Django 2.0 support

0.4

* ``upsert`` method
* Django 1.9 database functions
* ``bulk_update`` supports additional fields

0.3

* Lookup methods and operators
* ``F`` expressions and aggregation methods
* Database functions
* Conditional expressions for updates and annotations
* Bulk updates and change detection

0.2

* Change detection
* Groupby functionality
* Named tuples

.. _django: https://docs.djangoproject.com
.. _documentation: http://django-model-values.readthedocs.io
