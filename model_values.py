import collections
import functools
import itertools
import math
import operator
import types
import django
from django.db import IntegrityError, models, transaction
from django.db.models import functions
import six

map = six.moves.map
try:  # pragma: no cover
    import django.contrib.gis.db.models.functions
    import django.contrib.gis.db.models as gis
except Exception:  # pragma: no cover
    gis = None

try:
    from typing import Mapping
except ImportError:  # pragma: no cover
    from collections import Mapping

__version__ = '1.1'


def update_wrapper(wrapper, name):
    wrapper.__name__ = wrapper.__doc__ = name
    return wrapper


def method(lookup):
    return update_wrapper(lambda self, value: self.__eq__(value, '__' + lookup), lookup)


def starmethod(lookup):
    return update_wrapper(lambda self, *values: self.__eq__(values, '__' + lookup), lookup)


class Lookup(object):
    """Mixin for field lookups."""

    __ne__ = method('ne')
    __lt__ = method('lt')
    __le__ = method('lte')
    __gt__ = method('gt')
    __ge__ = method('gte')
    iexact = method('iexact')
    icontains = method('icontains')
    startswith = method('startswith')
    istartswith = method('istartswith')
    endswith = method('endswith')
    iendswith = method('iendswith')
    regex = method('regex')
    iregex = method('iregex')
    isin = method('in')
    range = starmethod('range')
    # spatial lookups
    contained = method('contained')
    coveredby = method('coveredby')
    covers = method('covers')
    crosses = method('crosses')
    disjoint = method('disjoint')
    equals = method('equals')  # __eq__ is taken
    intersects = method('intersects')  # __and__ is ambiguous
    relate = starmethod('relate')
    touches = method('touches')
    __lshift__ = left = method('left')
    __rshift__ = right = method('right')
    above = method('strictly_above')
    below = method('strictly_below')

    @property
    def is_valid(self):
        """Whether field `isvalid`."""
        return self.__eq__(True, '__isvalid')

    def contains(self, value, properly=False, bb=False):
        """Return whether field `contains` the value.  Options apply only to geom fields.

        :param properly: `contains_properly`
        :param bb: bounding box, `bbcontains`
        """
        return self.__eq__(value, '__{}contains{}'.format('bb' * bool(bb), '_properly' * bool(properly)))

    def overlaps(self, geom, position='', bb=False):
        """Return whether field `overlaps` with geometry .

        :param position: `overlaps_{left, right, above, below}`
        :param bb: bounding box, `bboverlaps`
        """
        return self.__eq__(geom, '__{}overlaps_{}'.format('bb' * bool(bb), position).rstrip('_'))

    def within(self, geom, distance=None):
        """Return whether field is `within` geometry.

        :param distance: `dwithin`
        """
        if distance is None:
            return self.__eq__(geom, '__within')
        return self.__eq__((geom, distance), '__dwithin')


class method(functools.partial):
    def __init__(self, func, *args):
        self.__doc__ = func.__doc__ or func.__name__

    def __get__(self, instance, owner):
        return self if instance is None else types.MethodType(self, instance)


def binary(func):
    return method(func), lambda *args: func(*args[::-1])


def transform(lookup, func, value):
    field, expr = func.source_expressions
    expr = expr if isinstance(expr, models.F) else expr.value
    return field.__eq__((expr, value), '__' + lookup)


class MetaF(type):
    def __getattr__(cls, name):
        if name in ('name', '__slots__'):
            raise AttributeError("'{}' is a reserved attribute".format(name))
        return cls(name)

    def any(cls, exprs):
        """Return ``Q`` OR object."""
        return functools.reduce(operator.or_, exprs)

    def all(cls, exprs):
        """Return ``Q`` AND object."""
        return functools.reduce(operator.and_, exprs)


class F(six.with_metaclass(MetaF, models.F, Lookup)):
    """Create ``F``, ``Q``, and ``Func`` objects with expressions.

    ``F`` creation supported as attributes:
    ``F.user`` == ``F('user')``,
    ``F.user.created`` == ``F('user__created')``.

    ``Q`` lookups supported as methods or operators:
    ``F.text.iexact(...)`` == ``Q(text__iexact=...)``,
    ``F.user.created >= ...`` == ``Q(user__created__gte=...)``.

    ``Func`` objects also supported as methods:
    ``F.user.created.min()`` == ``Min('user__created')``.
    """

    lookups = dict(length=functions.Length, lower=functions.Lower, upper=functions.Upper)
    coalesce = method(functions.Coalesce)
    concat = method(functions.Concat)  # __add__ is taken
    min = method(models.Min)
    max = method(models.Max)
    sum = method(models.Sum)
    mean = method(models.Avg)
    var = method(models.Variance)
    std = method(models.StdDev)
    greatest = method(functions.Greatest)
    least = method(functions.Least)
    now = staticmethod(functions.Now)
    cast = method(functions.Cast)
    extract = method(functions.Extract)
    trunc = method(functions.Trunc)
    if django.VERSION >= (2,):
        cume_dist = method(functions.CumeDist)
        dense_rank = method(functions.DenseRank)
        first_value = method(functions.FirstValue)
        lag = method(functions.Lag)
        last_value = method(functions.LastValue)
        lead = method(functions.Lead)
        nth_value = method(functions.NthValue)
        ntile = staticmethod(functions.Ntile)
        percent_rank = method(functions.PercentRank)
        rank = method(functions.Rank)
        row_number = method(functions.RowNumber)
        if gis:  # pragma: no cover
            azimuth = method(gis.functions.Azimuth)
            line_locate_point = method(gis.functions.LineLocatePoint)
    if django.VERSION >= (2, 1):
        lookups.update(chr=functions.Chr, ord=functions.Ord)
        strip = method(functions.Trim)
        lstrip = method(functions.LTrim)
        rstrip = method(functions.RTrim)
        repeat = method(functions.Repeat)
        if gis:  # pragma: no cover
            force_polygon_cw = method(gis.functions.ForcePolygonCW)
    if django.VERSION >= (2, 2):
        nullif = method(functions.NullIf)
        __reversed__ = method(functions.Reverse)
        __abs__ = method(functions.Abs)
        acos = method(functions.ACos)
        asin = method(functions.ASin)
        atan = method(functions.ATan)
        atan2 = method(functions.ATan2)
        __ceil__ = method(functions.Ceil)
        cos = method(functions.Cos)
        cot = method(functions.Cot)
        degrees = method(functions.Degrees)
        exp = method(functions.Exp)
        __floor__ = method(functions.Floor)
        __mod__, __rmod__ = binary(functions.Mod)
        pi = functions.Pi()
        __pow__, __rpow__ = binary(functions.Power)
        radians = method(functions.Radians)
        __round__ = method(functions.Round)
        sin = method(functions.Sin)
        sqrt = method(functions.Sqrt)
        tan = method(functions.Tan)
    if django.VERSION >= (3,):
        sign = method(functions.Sign)
        md5 = method(functions.MD5)
        sha1 = method(functions.SHA1)
        sha224 = method(functions.SHA224)
        sha256 = method(functions.SHA256)
        sha384 = method(functions.SHA384)
        sha512 = method(functions.SHA512)
    if gis:  # pragma: no cover
        area = property(gis.functions.Area)
        geojson = method(gis.functions.AsGeoJSON)
        gml = method(gis.functions.AsGML)
        kml = method(gis.functions.AsKML)
        svg = method(gis.functions.AsSVG)
        bounding_circle = method(gis.functions.BoundingCircle)
        centroid = property(gis.functions.Centroid)
        difference = method(gis.functions.Difference)
        envelope = property(gis.functions.Envelope)
        geohash = method(gis.functions.GeoHash)  # __hash__ requires an int
        intersection = method(gis.functions.Intersection)
        make_valid = method(gis.functions.MakeValid)
        mem_size = property(gis.functions.MemSize)
        num_geometries = property(gis.functions.NumGeometries)
        num_points = property(gis.functions.NumPoints)
        perimeter = property(gis.functions.Perimeter)
        point_on_surface = property(gis.functions.PointOnSurface)
        reverse = method(gis.functions.Reverse)
        scale = method(gis.functions.Scale)
        snap_to_grid = method(gis.functions.SnapToGrid)
        symmetric_difference = method(gis.functions.SymDifference)
        transform = method(gis.functions.Transform)
        translate = method(gis.functions.Translate)
        union = method(gis.functions.Union)

        @method
        class distance(gis.functions.Distance):
            """Return ``Distance`` with support for lookups: <, <=, >, >=, within."""

            __lt__ = method(transform, 'distance_lt')
            __le__ = method(transform, 'distance_lte')
            __gt__ = method(transform, 'distance_gt')
            __ge__ = method(transform, 'distance_gte')
            within = method(transform, 'dwithin')

    def __getattr__(self, name):
        """Return new `F`_ object with chained attribute."""
        return type(self)('{}__{}'.format(self.name, name))

    def __eq__(self, value, lookup=''):
        """Return ``Q`` object with lookup."""
        return models.Q(**{self.name + lookup: value})

    def __call__(self, *args, **extra):
        name, _, func = self.name.rpartition('__')
        return self.lookups[func](name, *args, **extra)

    def __getitem__(self, slc):
        """Return field ``Substr`` or ``Right``."""
        assert (slc.stop or 0) >= 0 and slc.step is None
        start = slc.start or 0
        if start < 0:
            assert slc.stop is None
            return functions.Right(self, -start)
        size = slc.stop and max(slc.stop - start, 0)
        return functions.Substr(self, start + 1, size)

    @method
    def count(self='*', **extra):
        """Return ``Count`` with optional field."""
        return models.Count(getattr(self, 'name', self), **extra)

    def find(self, sub, **extra):
        """Return ``StrIndex`` with ``str.find`` semantics."""
        return functions.StrIndex(self, Value(sub), **extra) - 1

    def replace(self, old, new='', **extra):
        """Return ``Replace`` with wrapped values."""
        return functions.Replace(self, Value(old), Value(new), **extra)

    def ljust(self, width, fill=' ', **extra):
        """Return ``LPad`` with wrapped values."""
        return functions.LPad(self, width, Value(fill), **extra)

    def rjust(self, width, fill=' ', **extra):
        """Return ``RPad`` with wrapped values."""
        return functions.RPad(self, width, Value(fill), **extra)

    def log(self, base=math.e, **extra):
        """Return ``Log``, by default ``Ln``."""
        return functions.Log(self, base, **extra)


def method(func):
    return update_wrapper(lambda self: self.reduce(func), func.__name__)


def binary(func):
    return update_wrapper(lambda self, value: func(models.F(*self._fields), value), func.__name__)


class QuerySet(models.QuerySet, Lookup):
    min = method(models.Min)
    max = method(models.Max)
    sum = method(models.Sum)
    mean = method(models.Avg)
    var = method(models.Variance)
    std = method(models.StdDev)
    __add__ = binary(operator.add)
    __sub__ = binary(operator.sub)
    __mul__ = binary(operator.mul)
    __truediv__ = __div__ = binary(operator.truediv)
    __mod__ = binary(operator.mod)
    __pow__ = binary(operator.pow)
    if gis:  # pragma: no cover
        collect = method(gis.Collect)
        extent = method(gis.Extent)
        extent3d = method(gis.Extent3D)
        make_line = method(gis.MakeLine)
        union = method(gis.Union)

    @property
    def _flat(self):
        return issubclass(self._iterable_class, models.query.FlatValuesListIterable)

    @property
    def _named(self):
        return issubclass(self._iterable_class, getattr(models.query, 'NamedValuesListIterable', ()))

    def __getitem__(self, key):
        """Allow column access by field names, expressions, or ``F`` objects.

        ``qs[field]`` returns flat ``values_list``

        ``qs[field, ...]`` returns tupled ``values_list``

        ``qs[Q_obj]`` provisionally returns filtered `QuerySet`_
        """
        if isinstance(key, tuple):
            kwargs = {'named': True} if django.VERSION >= (2,) else {}
            return self.values_list(*map(extract, key), **kwargs)
        key = extract(key)
        if isinstance(key, six.string_types + (models.Expression,)):
            return self.values_list(key, flat=True)
        if isinstance(key, models.Q):
            return self.filter(key)
        return super(QuerySet, self).__getitem__(key)

    def __setitem__(self, key, value):
        """Update a single column."""
        self.update(**{key: value})

    def __eq__(self, value, lookup=''):
        """Return `QuerySet`_ filtered by comparison to given value."""
        (field,) = self._fields
        return self.filter(**{field + lookup: value})

    def __contains__(self, value):
        """Return whether value is present using ``exists``."""
        if self._result_cache is None and self._flat:
            return (self == value).exists()
        return value in iter(self)

    def __iter__(self):
        """Iteration extended to support :meth:`groupby`."""
        if not hasattr(self, '_groupby'):
            return super(QuerySet, self).__iter__()
        size = len(self._groupby)
        rows = self[self._groupby + self._fields].order_by(*self._groupby).iterator()
        groups = itertools.groupby(rows, key=operator.itemgetter(*range(size)))
        getter = operator.itemgetter(size if self._flat else slice(size, None))
        if self._named:
            Row = collections.namedtuple('Row', self._fields)
            getter = lambda tup: Row(*tup[size:])  # noqa
        return ((key, map(getter, values)) for key, values in groups)

    def items(self, *fields, **annotations):
        """Return annotated ``values_list``."""
        return self.annotate(**annotations)[fields + tuple(annotations)]

    def groupby(self, *fields, **annotations):
        """Return a grouped `QuerySet`_.

        The queryset is iterable in the same manner as ``itertools.groupby``.
        Additionally the :meth:`reduce` functions will return annotated querysets.
        """
        qs = self.annotate(**annotations)
        qs._groupby = fields + tuple(annotations)
        return qs

    def annotate(self, *args, **kwargs):
        """Annotate extended to also handle mapping values, as a `Case`_ expression.

        :param kwargs: ``field={Q_obj: value, ...}, ...``

        As a provisional feature, an optional ``default`` key may be specified.
        """
        for field, value in kwargs.items():
            if Case.isa(value):
                kwargs[field] = Case.defaultdict(value)
        return super(QuerySet, self).annotate(*args, **kwargs)

    def value_counts(self, alias='count'):
        """Return annotated value counts."""
        return self.items(*self._fields, **{alias: F.count()})

    def sort_values(self, reverse=False):
        """Return `QuerySet`_ ordered by selected values."""
        qs = self.order_by(*self._fields)
        return qs.reverse() if reverse else qs

    def reduce(self, *funcs):
        """Return aggregated values, or an annotated `QuerySet`_ if :meth:`groupby` is in use.

        :param funcs: aggregation function classes
        """
        funcs = [func(field) for field, func in zip(self._fields, itertools.cycle(funcs))]
        if hasattr(self, '_groupby'):
            return self[self._groupby].annotate(*funcs)
        names = [func.default_alias for func in funcs]
        row = self.aggregate(*funcs)
        if self._named:
            return collections.namedtuple('Row', names)(**row)
        return row[names[0]] if self._flat else tuple(map(row.__getitem__, names))

    def update(self, **kwargs):
        """Update extended to also handle mapping values, as a `Case`_ expression.

        :param kwargs: ``field={Q_obj: value, ...}, ...``
        """
        for field, value in kwargs.items():
            if Case.isa(value):
                kwargs[field] = Case(value, default=F(field))
        return super(QuerySet, self).update(**kwargs)

    def change(self, defaults={}, **kwargs):
        """Update and return number of rows that actually changed.

        For triggering on-change logic without fetching first.

        ``if qs.change(status=...):`` status actually changed

        ``qs.change({'last_modified': now}, status=...)`` last_modified only updated if status updated

        :param defaults: optional mapping which will be updated conditionally, as with ``update_or_create``.
        """
        return self.exclude(**kwargs).update(**dict(defaults, **kwargs))

    def changed(self, **kwargs):
        """Return first mapping of fields and values which differ in the db.

        Also efficient enough to be used in boolean contexts, instead of ``exists``.
        """
        row = self.exclude(**kwargs).values(*kwargs).first() or {}
        return {field: value for field, value in row.items() if value != kwargs[field]}

    def exists(self, count=1):
        """Return whether there are at least the specified number of rows."""
        if count == 1:
            return super(QuerySet, self).exists()
        return (self[:count].count() if self._result_cache is None else len(self)) >= count


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

    def build_lookup(self, lookups, lhs, rhs):
        if rhs is None and lookups[-1:] == ['ne']:
            rhs, lookups[-1] = False, 'isnull'
        return super(Query, self).build_lookup(lookups, lhs, rhs)


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
        """Return whether primary key is present using ``exists``."""
        return self[pk].exists()

    def upsert(self, defaults={}, **kwargs):
        """Update or insert returning number of rows or created object.

        Faster and safer than ``update_or_create``.
        Supports combined expression updates by assuming the identity element on insert:  ``F(...) + 1``.

        :param defaults: optional mapping which will be updated, as with ``update_or_create``.
        """
        update = getattr(self.filter(**kwargs), 'update' if defaults else 'count')
        for field, value in defaults.items():
            expr = isinstance(value, models.expressions.CombinedExpression)
            kwargs[field] = value.rhs.value if expr else value
        try:
            with transaction.atomic():
                return update(**defaults) or self.create(**kwargs)
        except IntegrityError:
            return update(**defaults)

    def bulk_changed(self, field, data, key='pk'):
        """Return mapping of values which differ in the db.

        :param field: value column
        :param data: ``{pk: value, ...}``
        :param key: unique key column
        """
        rows = self.filter(F(key).isin(data))[key, field].iterator()
        return {pk: value for pk, value in rows if value != data[pk]}

    def bulk_change(self, field, data, key='pk', conditional=False, **kwargs):
        """Update changed rows with a minimal number of queries, by inverting the data to use ``pk__in``.

        :param field: value column
        :param data: ``{pk: value, ...}``
        :param key: unique key column
        :param conditional: execute select query and single conditional update;
            may be more efficient if the percentage of changed rows is relatively small
        :param kwargs: additional fields to be updated
        """
        if conditional:
            data = {pk: data[pk] for pk in self.bulk_changed(field, data, key)}
        updates = collections.defaultdict(list)
        for pk in data:
            updates[data[pk]].append(pk)
        if conditional:
            kwargs[field] = {F(key).isin(tuple(updates[value])): value for value in updates}
            return self.filter(F(key).isin(data)).update(**kwargs)
        count = 0
        for value in updates:
            kwargs[field] = value
            count += self.filter((F(field) != value) & F(key).isin(updates[value])).update(**kwargs)
        return count


class classproperty(property):
    """A property bound to a class."""

    def __get__(self, instance, owner):
        return self.fget(owner)


def Value(value):
    return value if isinstance(value, models.F) else models.Value(value)


def extract(field):
    if isinstance(field, models.F):
        return field.name
    return Case.defaultdict(field) if Case.isa(field) else field


class Case(models.Case):
    """``Case`` expression from mapping of when conditionals.

    :param conds: ``{Q_obj: value, ...}``
    :param default: optional default value or ``F`` object
    :param output_field: optional field defaults to registered ``types``
    """

    types = {
        str: models.CharField,
        int: models.IntegerField,
        float: models.FloatField,
        bool: models.BooleanField,
    }

    def __init__(self, conds, default=None, **extra):
        cases = (models.When(cond, Value(conds[cond])) for cond in conds)
        types = set(map(type, conds.values()))
        if len(types) == 1 and types.issubset(self.types):
            extra.setdefault('output_field', self.types.get(*types)())
        super(Case, self).__init__(*cases, default=Value(default), **extra)

    @classmethod
    def defaultdict(cls, conds):
        conds = dict(conds)
        return cls(conds, default=conds.pop('default', None))

    @classmethod
    def isa(cls, value):
        return isinstance(value, Mapping) and any(isinstance(key, models.Q) for key in value)


def EnumField(enum, display=None, **options):
    """Return a ``CharField`` or ``IntegerField`` with choices from given enum.

    By default, enum names and values are used as db values and display labels respectively,
    returning a ``CharField`` with computed ``max_length``.

    :param display: optional callable to transform enum names to display labels,
         thereby using enum values as db values and also supporting integers.
    """
    choices = tuple((choice.name, choice.value) for choice in enum)
    if display is not None:
        choices = tuple((choice.value, display(choice.name)) for choice in enum)
    try:
        max_length = max(map(len, dict(choices)))
    except TypeError:
        return models.IntegerField(choices=choices, **options)
    return models.CharField(max_length=max_length, choices=choices, **options)
