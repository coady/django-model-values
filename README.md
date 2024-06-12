[![image](https://img.shields.io/pypi/v/django-model-values.svg)](https://pypi.org/project/django-model-values/)
![image](https://img.shields.io/pypi/pyversions/django-model-values.svg)
![image](https://img.shields.io/pypi/djversions/django-model-values.svg)
[![image](https://pepy.tech/badge/django-model-values)](https://pepy.tech/project/django-model-values)
![image](https://img.shields.io/pypi/status/django-model-values.svg)
[![build](https://github.com/coady/django-model-values/actions/workflows/build.yml/badge.svg)](https://github.com/coady/django-model-values/actions/workflows/build.yml)
[![image](https://codecov.io/gh/coady/django-model-values/branch/main/graph/badge.svg)](https://codecov.io/gh/coady/django-model-values/)
[![CodeQL](https://github.com/coady/django-model-values/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/coady/django-model-values/actions/workflows/github-code-scanning/codeql)
[![image](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![image](https://mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

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
