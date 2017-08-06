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
   :exclude-members: __or__

   .. note:: Spatial functions are experimental and may change in the future.
      See source for available functions if gis is configured.

   .. method:: __or__

      .. deprecated:: 0.5
         Replaced by gis ``union``; use ``coalesce`` instead.

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

classproperty
=============
.. autoclass:: classproperty
   :show-inheritance:
