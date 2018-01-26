.. automodule:: model_values

Lookup
=============
.. autoclass:: Lookup
   :members:
   :special-members:
   :exclude-members: __weakref__

   .. note:: Spatial lookups require `gis`_ to be enabled.

F
=============
.. autoclass:: F
   :show-inheritance:
   :members:
   :special-members:

   .. note:: Since attributes are used for constructing `F`_ objects, there may be collisions between field names and methods.
      For example, ``name`` is a reserved attribute, but the usual constructor can still be used: ``F('name')``.

   .. note:: See source for available spatial functions if `gis`_ is configured.

   .. autoattribute:: lookups

      mapping of potentially `registered lookups`_ to transform functions

QuerySet
=============
.. autoclass:: QuerySet
   :show-inheritance:
   :members:
   :special-members:

   .. note:: See source for available aggregate spatial functions if `gis`_ is configured.

Manager
=============
.. autoclass:: Manager
   :show-inheritance:
   :members:
   :special-members:

Case
=============
.. autoclass:: Case
   :show-inheritance:
   :members:

   .. autoattribute:: types

      mapping of types to output fields

classproperty
=============
.. autoclass:: classproperty
   :show-inheritance:

.. _`registered lookups`: https://docs.djangoproject.com/en/stable/ref/models/database-functions/#length
.. _`gis`: https://docs.djangoproject.com/en/stable/ref/contrib/gis/
