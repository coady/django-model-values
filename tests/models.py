from django.db import models
from model_values import Manager


class Book(models.Model):
    title = models.TextField()
    author = models.CharField(max_length=50)
    quantity = models.IntegerField()
    last_modified = models.DateTimeField(auto_now=True)

    objects = Manager()
