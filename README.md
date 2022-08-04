[![image](https://img.shields.io/pypi/v/django-model-values.svg)](https://pypi.org/project/django-model-values/)
![image](https://img.shields.io/pypi/pyversions/django-model-values.svg)
![image](https://img.shields.io/pypi/djversions/django-model-values.svg)
[![image](https://pepy.tech/badge/django-model-values)](https://pepy.tech/project/django-model-values)
![image](https://img.shields.io/pypi/status/django-model-values.svg)
[![image](https://github.com/coady/django-model-values/workflows/build/badge.svg)](https://github.com/coady/django-model-values/actions)
[![image](https://codecov.io/gh/coady/django-model-values/branch/main/graph/badge.svg)](https://codecov.io/gh/coady/django-model-values/)
[![image](https://github.com/coady/django-model-values/workflows/codeql/badge.svg)](https://github.com/coady/django-model-values/security/code-scanning)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://pypi.org/project/black/)
[![image](http://mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

[Django](https://docs.djangoproject.com) model utilities for encouraging direct data access instead of unnecessary object overhead. Implemented through compatible method and operator extensions to `QuerySets` and `Managers`.

The goal is to provide elegant syntactic support for best practices in using Django's ORM. Specifically avoiding the inefficiencies and race conditions associated with always using objects.

## Usage
Typical model usage is verbose, inefficient, and incorrect.

```python
book = Book.objects.get(pk=pk)
book.rating = 5.0
book.save()
```

The correct method is generally supported, but arguably less readable.

```python
Book.objects.filter(pk=pk).update(rating=5.0)
```

`model_values` encourages the better approach with operator support.

```python
Book.objects[pk]['rating'] = 5.0
```

Similarly for queries:

```python
(book.rating for book in books)
books.values_list('rating', flat=True)
books['rating']
```

Column-oriented syntax is common in panel data layers, and the greater expressiveness cascades. `QuerySets` also support aggregation and conditionals.

```python
books.values_list('rating', flat=True).filter(rating__gt=0)
books['rating'] > 0

books.aggregate(models.Avg('rating'))['rating__avg']
books['rating'].mean()
```

`Managers` provide a variety of efficient primary key based utilities. To enable, instantiate the `Manager` in your models. As with any custom `Manager`, it doesn't have to be named `objects`, but it is designed to be a 100% compatible replacement.

```python
from model_values import Manager

class Book(models.Model):
    ...
    objects = Manager()
```

`F` expressions are also enhanced, and can be used directly without model changes.

```python
from model_values import F

.filter(rating__gt=0, last_modified__range=(start, end))
.filter(F.rating > 0, F.last_modified.range(start, end))
```

## Installation
```console
% pip install django-model-values
```

## Tests
100% branch coverage.

```console
% pytest [--cov]
```

## Changes
1.5
* Django >=3.2 required

1.4
* Python >=3.7 required
* Django 4 support

1.3
* Django 3.2 support

1.2
* Python >=3.6 required
* Django >=2.2 required

1.1
* Django 3 support

1.0
* Update related methods moved with deprecation warnings
* Extensible change detection and updates
* Django 2.2 functions

0.6
* Transform functions
* Named tuples
* Window functions
* Distance lookups
* Django 2.1 functions
* `EnumField`
* Annotated `items`
* Expressions in column selection

0.5
* `F` expressions operators `any` and `all`
* Spatial lookups and functions
* Django 2.0 support

0.4
* `upsert` method
* Django 1.9 database functions
* `bulk_update` supports additional fields

0.3
* Lookup methods and operators
* `F` expressions and aggregation methods
* Database functions
* Conditional expressions for updates and annotations
* Bulk updates and change detection

0.2
* Change detection
* Groupby functionality
* Named tuples
