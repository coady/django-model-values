Example
===========
An example ``Model`` used in the tests.

.. literalinclude:: ../tests/models.py
   :lines: 1-11

Table logic
^^^^^^^^^^^
Django recommends model methods for row-level functionality,
and `custom managers`_ for table-level functionality.
That's fine if the custom managers are reused across models,
but often they're just custom filters, and specific to a model.
As evidenced by `django-model-utils'`_ ``QueryManager`` and ``PassThroughManager``.

There's a simpler way to achieve the same end: a model ``classmethod``.
In some cases a profileration of classmethods is an anti-pattern, but in this case functions won't suffice.
It's Django that attached the ``Manager`` instance to a class.

Additionally a ``classproperty`` wrapper is provided,
to mimic a custom ``Manager`` or ``Queryset`` without calling it first.

.. literalinclude:: ../tests/models.py
   :lines: 13-15

Row logic
^^^^^^^^^^^
Some of the below methods may be added to a model mixin in the future.
It's a delicate balance, as the goal is to *not* encourage object usage.
However, sometimes having an object already is inevitable,
so it's still worth considering best practices given that situation.

Providing wrappers for any manager method that's ``pk``-based may be worthwhile,
particularly a filter to match only the object.

.. literalinclude:: ../tests/models.py
   :lines: 17-19

From there one can easily imagine other useful extensions.

.. literalinclude:: ../tests/models.py
   :lines: 21-

.. _`custom managers`: https://docs.djangoproject.com/en/1.9/topics/db/managers/#custom-managers
.. _`django-model-utils'`: https://pypi.python.org/pypi/django-model-utils/
