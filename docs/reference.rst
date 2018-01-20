.. automodule:: model_values

Lookup
=============
.. autoclass:: Lookup
   :members:
   :special-members:
   :exclude-members: __weakref__

   .. note:: Spatial lookups are experimental and may change in the future.

F
=============
.. autoclass:: F
   :show-inheritance:
   :members:
   :special-members:

   .. note:: Spatial functions are experimental and may change in the future.
      See source for available functions if gis is configured.

   .. autoattribute:: lookups

      mapping of potentially `registered lookups`_ to transform functions

QuerySet
=============
.. autoclass:: QuerySet
   :show-inheritance:
   :members:
   :special-members:

   .. note:: Spatial aggregate functions are experimental and may change in the future.
      See source for available functions if gis is configured.

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
