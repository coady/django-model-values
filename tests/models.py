from django.db import models
from model_values import F, Manager, classproperty


class Book(models.Model):
    title = models.TextField()
    author = models.CharField(max_length=50)
    quantity = models.IntegerField()
    last_modified = models.DateTimeField(auto_now=True)

    objects = Manager()

    @classproperty
    def in_stock(cls):
        return cls.objects.filter(F.quantity > 0)

    @property
    def object(self):
        return type(self).objects[self.pk]

    def changed(self, **kwargs):
        return self.object.changed(**kwargs)

    def update(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])
        return self.object.update(**kwargs)
