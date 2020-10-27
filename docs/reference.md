::: model_values.Lookup
!!! note
    Spatial lookups require [gis](https://docs.djangoproject.com/en/stable/ref/contrib/gis/) to be enabled.

::: model_values.F
!!! note
    Since attributes are used for constructing [F](#model_values.F) objects, there may be collisions between field names and methods. For example, `name` is a reserved attribute, but the usual constructor can still be used: `F('name')`.
!!! note
    See source for available spatial functions if [gis](https://docs.djangoproject.com/en/stable/ref/contrib/gis/) is configured.

::: model_values.QuerySet
!!! note
    See source for available aggregate spatial functions if [gis](https://docs.djangoproject.com/en/stable/ref/contrib/gis/) is configured.

::: model_values.Manager

::: model_values.Case

::: model_values.classproperty

::: model_values.EnumField
