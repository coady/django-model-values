An example `Model` used in the tests.

```python
from django.db import models
from model_values import F, Manager, classproperty


class Book(models.Model):
    title = models.TextField()
    author = models.CharField(max_length=50)
    quantity = models.IntegerField()
    last_modified = models.DateTimeField(auto_now=True)

    objects = Manager()
```

## Table logic

Django recommends model methods for row-level functionality, and [custom managers](https://docs.djangoproject.com/en/stable/topics/db/managers/#custom-managers) for table-level functionality. That's fine if the custom managers are reused across models, but often they're just custom filters, and specific to a model. As evidenced by [django-model-utils'](https://pypi.org/project/django-model-utils/) `QueryManager`.

There's a simpler way to achieve the same end: a model `classmethod`. In some cases a profileration of classmethods is an anti-pattern, but in this case functions won't suffice. It's Django that attached the `Manager` instance to a class.

Additionally a `classproperty` wrapper is provided, to mimic a custom `Manager` or `Queryset` without calling it first.

```python
    @classproperty
    def in_stock(cls):
        return cls.objects.filter(F.quantity > 0)
```

## Row logic

Some of the below methods may be added to a model mixin in the future. It's a delicate balance, as the goal is to *not* encourage object usage. However, sometimes having an object already is inevitable, so it's still worth considering best practices given that situation.

Providing wrappers for any manager method that's `pk`-based may be worthwhile, particularly a filter to match only the object.

```python
    @property
    def object(self):
        return type(self).objects[self.pk]
```

From there one can easily imagine other useful extensions.

```python
    def changed(self, **kwargs):
        return self.object.changed(**kwargs)

    def update(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])
        return self.object.update(**kwargs)
```
