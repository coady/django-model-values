import collections
import itertools
import operator
import django
from django.db import IntegrityError, models, transaction
from django.db.models import functions
from django.utils import six
map = six.moves.map

__version__ = '0.4'


def update_wrapper(wrapper, name):
    wrapper.__name__ = wrapper.__doc__ = name
    return wrapper


def method(lookup):
    return update_wrapper(lambda self, value: self.__eq__(value, '__' + lookup), lookup)


class Lookup(object):
    """Mixin for field lookups."""
    __ne__ = method('ne')
    __lt__ = method('lt')
    __le__ = method('lte')
    __gt__ = method('gt')
    __ge__ = method('gte')
    iexact = method('iexact')
    contains = method('contains')  # __contains__ coerces to a bool
    icontains = method('icontains')
    startswith = method('startswith')
    istartswith = method('istartswith')
    endswith = method('endswith')
    iendswith = method('iendswith')
    search = method('search')
    regex = method('regex')
    iregex = method('iregex')

    def in_(self, *values):
        """in"""
        return self.__eq__(values, '__in')

    def range(self, *values):
        """range"""
        return self.__eq__(values, '__range')


def method(func):
    return update_wrapper(lambda *args, **extra: func(*args, **extra), func.__name__)


class FExpr(models.F, Lookup):
    """Singleton for creating ``F``, ``Q``, ``Func``, and ``OrderBy`` objects with expressions.

    ``F.user.created`` == ``F('user__created')``

    ``F.user.created >= ...`` == ``Q(user__created__gte=...)``

    ``F.user.created.min()`` == ``Min('user__created')``

    ``-F.user.created`` == ``F('user__created').desc()``

    ``F.text.iexact(...)`` == ``Q(text__iexact=...)``
    """
    __neg__ = models.F.desc
    __or__ = method(functions.Coalesce)
    concat = method(functions.Concat)  # __add__ is taken
    length = method(functions.Length)  # __len__ requires an int
    lower = method(functions.Lower)
    upper = method(functions.Upper)
    min = method(models.Min)
    max = method(models.Max)
    sum = method(models.Sum)
    mean = method(models.Avg)
    count = method(models.Count)
    var = method(models.Variance)
    std = method(models.StdDev)
    if django.VERSION >= (1, 9):
        greatest = method(functions.Greatest)
        least = method(functions.Least)
        now = staticmethod(functions.Now)

    def __getattr__(self, name):
        """Return new `F`_ object with chained attribute."""
        return type(self)('{}__{}'.format(self.name, name).lstrip('_'))
    __call__ = __getattr__

    def __eq__(self, value, lookup=''):
        """Return ``Q`` object with lookup."""
        return models.Q(**{self.name + lookup: value})

    def __getitem__(self, slc):
        """Return field ``Substr``."""
        start = slc.start or 0
        size = slc.stop and slc.stop - start
        assert start >= 0 and (size is None or size >= 0) and slc.step is None
        return functions.Substr(self, start + 1, size)
F = FExpr('')


def method(func):
    return update_wrapper(lambda self: self.reduce(func), func.__name__)


class QuerySet(models.QuerySet, Lookup):
    min = method(models.Min)
    max = method(models.Max)
    sum = method(models.Sum)
    mean = method(models.Avg)
    var = method(models.Variance)
    std = method(models.StdDev)

    @property
    def _flat(self):
        return issubclass(self._iterable_class, models.query.FlatValuesListIterable)

    @_flat.setter
    def _flat(self, value):
        classes = models.query.FlatValuesListIterable, models.query.ValuesListIterable
        if issubclass(self._iterable_class, classes):
            self._iterable_class = classes[not value]

    if django.VERSION < (1, 9):
        _flat = property(lambda self: getattr(self, 'flat', None),
                         lambda self, value: setattr(self, 'flat', bool(value)))

    def __getitem__(self, key):
        """Allow column access by field names (or ``F`` objects) and filtering by ``Q`` objects.

        ``qs[field]`` returns flat ``values_list``

        ``qs[field, ...]`` returns tupled ``values_list``

        ``qs[Q_obj]`` returns filtered `QuerySet`_
        """
        if isinstance(key, tuple):
            fields = (field.name if isinstance(field, models.F) else field for field in key)
            return self.values_list(*fields)
        if isinstance(key, six.string_types):
            return self.values_list(key, flat=True)
        if isinstance(key, models.F):
            return self.values_list(key.name, flat=True)
        if isinstance(key, models.Q):
            return self.filter(key)
        return super(QuerySet, self).__getitem__(key)

    def __setitem__(self, key, value):
        """Update a single column."""
        self.update(**{key: value})

    def __eq__(self, value, lookup=''):
        """Return `QuerySet`_ filtered by comparison to given value."""
        lookups = (field + lookup for field in self._fields)
        return self.filter(**dict.fromkeys(lookups, value))

    def __contains__(self, value):
        """Return whether value is present using ``exists``."""
        if self._result_cache is None and self._flat:
            return (self == value).exists()
        return value in iter(self)

    @property
    def F(self):
        return models.F(*self._fields)

    def __add__(self, value):
        """F + value."""
        return self.F + value

    def __sub__(self, value):
        """F - value."""
        return self.F - value

    def __mul__(self, value):
        """F * value."""
        return self.F * value

    def __truediv__(self, value):
        """F / value."""
        return self.F / value
    __div__ = __truediv__

    def __mod__(self, value):
        """F % value."""
        return self.F % value

    def __pow__(self, value):
        """F ** value."""
        return self.F ** value

    def __iter__(self):
        if not hasattr(self, '_groupby'):
            return super(QuerySet, self).__iter__()
        size = len(self._groupby)
        rows = self[self._groupby + self._fields].order_by(*self._groupby).iterator()
        groups = itertools.groupby(rows, key=operator.itemgetter(*range(size)))
        Values = collections.namedtuple('Values', self._fields)
        getter = operator.itemgetter(size) if self._flat else lambda tup: Values(*tup[size:])
        return ((key, map(getter, values)) for key, values in groups)

    def groupby(self, *fields, **annotations):
        """Return a grouped `QuerySet`_.

        The queryset is iterable in the same manner as ``itertools.groupby``.
        Additionally the ``reduce`` functions will return annotated querysets.
        """
        qs = self.annotate(**annotations)
        qs._groupby = fields + tuple(annotations)
        return qs

    def annotate(self, *args, **kwargs):
        """Annotate extended to also handle mapping values, as a case expression.

        :param kwargs: ``field={Q_obj: value, ...}, ...``
        """
        if hasattr(self, '_groupby'):
            self = self[self._groupby]
        kwargs.update((field, Case(value)) for field, value in kwargs.items()
                      if isinstance(value, collections.Mapping))
        qs = super(QuerySet, self).annotate(*args, **kwargs)
        if args or kwargs:
            qs._flat = False
        return qs

    def value_counts(self, alias='count'):
        """Return annotated value counts."""
        return self.annotate(**{alias: F.count()})

    def reduce(self, *funcs):
        """Return aggregated values, or an annotated `QuerySet`_ if ``groupby`` is in use.

        :param funcs: aggregation function classes
        """
        funcs = [func(field) for field, func in zip(self._fields, itertools.cycle(funcs))]
        if hasattr(self, '_groupby'):
            return self.annotate(*funcs)
        names = (func.default_alias for func in funcs)
        values = collections.namedtuple('Values', names)(**self.aggregate(*funcs))
        return values[0] if self._flat else values

    def update(self, **kwargs):
        """Update extended to also handle mapping values, as a case expression.

        :param kwargs: ``field={Q_obj: value, ...}, ...``
        """
        kwargs.update((field, Case(value, default=field)) for field, value in kwargs.items()
                      if isinstance(value, collections.Mapping))
        return super(QuerySet, self).update(**kwargs)

    def modify(self, defaults=(), **kwargs):
        """Update and return number of rows that actually changed.

        For triggering on-change logic without fetching first.

        ``if qs.modify(status=...):`` status actually changed

        ``qs.modify({'last_modified': now}, status=...)`` last_modified only updated if status updated

        :param defaults: optional mapping which will be updated conditionally, as with ``update_or_create``.
        """
        return self.exclude(**kwargs).update(**dict(defaults, **kwargs))

    def upsert(self, defaults={}, **kwargs):
        """Update or insert returning number of rows or created object; faster and safer than ``update_or_create``.

        Supports combined expression updates by assuming the identity element on insert:  ``F(...) + 1``.

        :param defaults: optional mapping which will be updated, as with ``update_or_create``.
        """
        lookup, params = self._extract_model_params(defaults, **kwargs)
        for field, value in params.items():
            if isinstance(value, models.expressions.CombinedExpression):
                params[field] = value.rhs.value
        update = getattr(self.filter(**lookup), 'update' if defaults else 'count')
        try:
            with transaction.atomic():
                return update(**defaults) or self.create(**params)
        except IntegrityError:
            return update(**defaults)

    def exists(self, count=1):
        """Return whether there are at least the specified number of rows."""
        if self._result_cache is None and count != 1:
            return len(self['pk'][:count]) >= count
        return super(QuerySet, self).exists()


@models.Field.register_lookup
class NotEqual(models.Lookup):
    """Missing != operator."""
    lookup_name = 'ne'

    def as_sql(self, *args):
        lhs, lhs_params = self.process_lhs(*args)
        rhs, rhs_params = self.process_rhs(*args)
        return '{} <> {}'.format(lhs, rhs), lhs_params + rhs_params


class Query(models.sql.Query):
    """Allow __ne=None lookup."""
    def prepare_lookup_value(self, value, lookups, *args):
        if value is None and lookups[-1:] == ['ne']:
            value, lookups[-1] = False, 'isnull'
        return super(Query, self).prepare_lookup_value(value, lookups, *args)


class Manager(models.Manager):
    def get_queryset(self):
        return QuerySet(self.model, Query(self.model), self._db, self._hints)

    def __getitem__(self, pk):
        """Return `QuerySet`_ which matches primary key.

        To encourage direct db access, instead of always using get and save.
        """
        return self.filter(pk=pk)

    def __delitem__(self, pk):
        """Delete row with primary key."""
        self[pk].delete()

    def __contains__(self, pk):
        """Return whether pk is present using ``exists``."""
        return self[pk].exists()

    def changed(self, pk, **kwargs):
        """Return mapping of fields and values which differ in the db.

        Also efficient enough to be used in boolean contexts, instead of ``exists``.
        """
        row = self[pk].exclude(**kwargs).values(*kwargs).first() or {}
        return {field: value for field, value in row.items() if value != kwargs[field]}

    def bulk_changed(self, field, data):
        """Return mapping of values which differ in the db.

        :param data: ``{pk: value, ...}``
        """
        rows = self.filter(pk__in=data)['pk', field].iterator()
        return {pk: value for pk, value in rows if value != data[pk]}

    def bulk_update(self, field, data, changed=False, conditional=False, **kwargs):
        """Update with a minimal number of queries, by inverting the data to use ``pk__in``.

        :param data: ``{pk: value, ...}``
        :param changed: execute select query first to update only rows which differ;
            more efficient if the expected percentage of changed rows is relatively small
        :param conditional: execute a single query with a conditional expression;
            may be more efficient if the number of rows is large (but bounded)
        :param kwargs: additional fields to be updated
        """
        if changed:
            data = {pk: data[pk] for pk in self.bulk_changed(field, data)}
        updates = collections.defaultdict(list)
        for pk in data:
            updates[data[pk]].append(pk)
        if conditional:
            kwargs[field] = {F.pk__in == updates[value]: value for value in updates}
            return self.filter(pk__in=data).update(**kwargs)
        count = 0
        for value in updates:
            kwargs[field] = value
            count += self.filter(pk__in=updates[value]).update(**kwargs)
        return count


class classproperty(property):
    """A property bound to a class."""
    def __get__(self, instance, owner):
        return self.fget(owner)


class Case(models.Case):
    """``Case`` expression from mapping of when conditionals."""
    types = {str: models.CharField, int: models.IntegerField, float: models.FloatField, bool: models.BooleanField}

    def __init__(self, conds, **extra):
        cases = (models.When(cond, models.Value(conds[cond])) for cond in conds)
        types = set(map(type, conds.values()))
        if 'default' not in extra and len(types) == 1 and types.issubset(self.types):
            extra.setdefault('output_field', self.types.get(*types)())
        super(Case, self).__init__(*cases, **extra)
