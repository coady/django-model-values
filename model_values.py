import collections
import functools
import itertools
import math
import operator
import types
from typing import Callable, Iterable, Mapping, Union
import django
from django.db import IntegrityError, models, transaction
from django.db.models import functions

try:  # pragma: no cover
    import django.contrib.gis.db.models.functions
    import django.contrib.gis.db.models as gis
except Exception:  # pragma: no cover
    gis = None

__version__ = '1.2'


def update_wrapper(wrapper, name):
    wrapper.__name__ = wrapper.__doc__ = name
    return wrapper


def eq(lookup):
    return update_wrapper(lambda self, value: self.__eq__(value, '__' + lookup), lookup)


class Lookup:
    """Mixin for field lookups."""

    __ne__ = eq('ne')
    __lt__ = eq('lt')
    __le__ = eq('lte')
    __gt__ = eq('gt')
    __ge__ = eq('gte')
    iexact = eq('iexact')
    icontains = eq('icontains')
    startswith = eq('startswith')
    istartswith = eq('istartswith')
    endswith = eq('endswith')
    iendswith = eq('iendswith')
    regex = eq('regex')
    iregex = eq('iregex')
    isin = eq('in')
    # spatial lookups
    contained = eq('contained')
    coveredby = eq('coveredby')
    covers = eq('covers')
    crosses = eq('crosses')
    disjoint = eq('disjoint')
    equals = eq('equals')  # __eq__ is taken
    intersects = eq('intersects')  # __and__ is ambiguous
    touches = eq('touches')
    __lshift__ = left = eq('left')
    __rshift__ = right = eq('right')
    above = eq('strictly_above')
    below = eq('strictly_below')

    def range(self, *values):
        """range"""
        return self.__eq__(values, '__range')

    def relate(self, *values):
        """relate"""
        return self.__eq__(values, '__relate')

    @property
    def is_valid(self):
        """Whether field `isvalid`."""
        return self.__eq__(True, '__isvalid')

    def contains(self, value, properly=False, bb=False):
        """Return whether field `contains` the value.  Options apply only to geom fields.

        :param properly: `contains_properly`
        :param bb: bounding box, `bbcontains`
        """
        properly = '_properly' * bool(properly)
        bb = 'bb' * bool(bb)
        return self.__eq__(value, f'__{bb}contains{properly}')

    def overlaps(self, geom, position='', bb=False):
        """Return whether field `overlaps` with geometry .

        :param position: `overlaps_{left, right, above, below}`
        :param bb: bounding box, `bboverlaps`
        """
        bb = 'bb' * bool(bb)
        return self.__eq__(geom, f'__{bb}overlaps_{position}'.rstrip('_'))

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


def transform(lookup, func, value):
    field, expr = func.source_expressions
    expr = expr if isinstance(expr, models.F) else expr.value
    return field.__eq__((expr, value), '__' + lookup)


class MetaF(type):
    def __getattr__(cls, name: str) -> 'F':
        if name in ('name', '__slots__'):
            raise AttributeError(f"'{name}' is a reserved attribute")
        return cls(name)

    def any(cls, exprs: Iterable[models.Q]) -> models.Q:
        """Return ``Q`` OR object."""
        return functools.reduce(operator.or_, exprs)

    def all(cls, exprs: Iterable[models.Q]) -> models.Q:
        """Return ``Q`` AND object."""
        return functools.reduce(operator.and_, exprs)


class F(models.F, Lookup, metaclass=MetaF):
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

    lookups = dict(
        length=functions.Length,
        lower=functions.Lower,
        upper=functions.Upper,
        chr=functions.Chr,
        ord=functions.Ord,
        acos=functions.ACos,
        asin=functions.ASin,
        atan=functions.ATan,
        atan2=functions.ATan2,
        cos=functions.Cos,
        cot=functions.Cot,
        degrees=functions.Degrees,
        exp=functions.Exp,
        radians=functions.Radians,
        sin=functions.Sin,
        sqrt=functions.Sqrt,
        tan=functions.Tan,
    )
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
    strip = method(functions.Trim)
    lstrip = method(functions.LTrim)
    rstrip = method(functions.RTrim)
    repeat = method(functions.Repeat)
    nullif = method(functions.NullIf)
    __reversed__ = method(functions.Reverse)
    __abs__ = method(functions.Abs)
    __ceil__ = method(functions.Ceil)
    __floor__ = method(functions.Floor)
    __mod__ = method(functions.Mod)
    pi = functions.Pi()
    __pow__ = method(functions.Power)
    __round__ = method(functions.Round)
    if django.VERSION >= (3,):
        lookups.update(sign=functions.Sign, md5=functions.MD5)
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
        azimuth = method(gis.functions.Azimuth)
        line_locate_point = method(gis.functions.LineLocatePoint)
        force_polygon_cw = method(gis.functions.ForcePolygonCW)

        @method
        class distance(gis.functions.Distance):
            """Return ``Distance`` with support for lookups: <, <=, >, >=, within."""

            __lt__ = method(transform, 'distance_lt')
            __le__ = method(transform, 'distance_lte')
            __gt__ = method(transform, 'distance_gt')
            __ge__ = method(transform, 'distance_gte')
            within = method(transform, 'dwithin')

    def __getattr__(self, name: str) -> 'F':
        """Return new `F`_ object with chained attribute."""
        return type(self)('{}__{}'.format(self.name, name))

    def __eq__(self, value, lookup: str = '') -> models.Q:
        """Return ``Q`` object with lookup."""
        if not lookup and type(value) is models.F:
            return self.name == value.name
        return models.Q(**{self.name + lookup: value})

    def __ne__(self, value) -> models.Q:
        """Allow __ne=None lookup without custom queryset."""
        if value is None:
            return self.__eq__(False, '__isnull')
        return self.__eq__(value, '__ne')

    __hash__ = models.F.__hash__

    def __call__(self, *args, **extra) -> models.Func:
        name, _, func = self.name.rpartition('__')
        return self.lookups[func](name, *args, **extra)

    def __iter__(self):
        raise TypeError("'F' object is not iterable")

    def __getitem__(self, slc: slice) -> models.Func:
        """Return field ``Substr`` or ``Right``."""
        assert (slc.stop or 0) >= 0 and slc.step is None
        start = slc.start or 0
        if start < 0:
            assert slc.stop is None
            return functions.Right(self, -start)
        size = slc.stop and max(slc.stop - start, 0)
        return functions.Substr(self, start + 1, size)

    def __rmod__(self, value):
        return functions.Mod(value, self)

    def __rpow__(self, value):
        return functions.Power(value, self)

    @method
    def count(self='*', **extra):
        """Return ``Count`` with optional field."""
        return models.Count(getattr(self, 'name', self), **extra)

    def find(self, sub, **extra) -> models.Expression:
        """Return ``StrIndex`` with ``str.find`` semantics."""
        return functions.StrIndex(self, Value(sub), **extra) - 1

    def replace(self, old, new='', **extra) -> models.Func:
        """Return ``Replace`` with wrapped values."""
        return functions.Replace(self, Value(old), Value(new), **extra)

    def ljust(self, width: int, fill=' ', **extra) -> models.Func:
        """Return ``LPad`` with wrapped values."""
        return functions.LPad(self, width, Value(fill), **extra)

    def rjust(self, width: int, fill=' ', **extra) -> models.Func:
        """Return ``RPad`` with wrapped values."""
        return functions.RPad(self, width, Value(fill), **extra)

    def log(self, base=math.e, **extra) -> models.Func:
        """Return ``Log``, by default ``Ln``."""
        return functions.Log(self, base, **extra)


def reduce(func):
    return update_wrapper(lambda self: self.reduce(func), func.__name__)


def field(func):
    return update_wrapper(lambda self, value: func(models.F(*self._fields), value), func.__name__)


class QuerySet(models.QuerySet, Lookup):
    min = reduce(models.Min)
    max = reduce(models.Max)
    sum = reduce(models.Sum)
    mean = reduce(models.Avg)
    var = reduce(models.Variance)
    std = reduce(models.StdDev)
    __add__ = field(operator.add)
    __sub__ = field(operator.sub)
    __mul__ = field(operator.mul)
    __truediv__ = field(operator.truediv)
    __mod__ = field(operator.mod)
    __pow__ = field(operator.pow)
    if gis:  # pragma: no cover
        collect = reduce(gis.Collect)
        extent = reduce(gis.Extent)
        extent3d = reduce(gis.Extent3D)
        make_line = reduce(gis.MakeLine)
        union = reduce(gis.Union)

    @property
    def _flat(self):
        return issubclass(self._iterable_class, models.query.FlatValuesListIterable)

    @property
    def _named(self):
        return issubclass(self._iterable_class, models.query.NamedValuesListIterable)

    def __getitem__(self, key):
        """Allow column access by field names, expressions, or ``F`` objects.

        ``qs[field]`` returns flat ``values_list``

        ``qs[field, ...]`` returns tupled ``values_list``

        ``qs[Q_obj]`` provisionally returns filtered `QuerySet`_
        """
        if isinstance(key, tuple):
            return self.values_list(*map(extract, key), named=True)
        key = extract(key)
        if isinstance(key, (str, models.Expression)):
            return self.values_list(key, flat=True)
        if isinstance(key, models.Q):
            return self.filter(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        """Update a single column."""
        self.update(**{key: value})

    def __eq__(self, value, lookup: str = '') -> 'QuerySet':
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
            return super().__iter__()
        size = len(self._groupby)
        rows = self[self._groupby + self._fields].order_by(*self._groupby).iterator()
        groups = itertools.groupby(rows, key=operator.itemgetter(*range(size)))
        getter = operator.itemgetter(size if self._flat else slice(size, None))
        if self._named:
            Row = collections.namedtuple('Row', self._fields)
            getter = lambda tup: Row(*tup[size:])  # noqa
        return ((key, map(getter, values)) for key, values in groups)

    def items(self, *fields, **annotations) -> 'QuerySet':
        """Return annotated ``values_list``."""
        return self.annotate(**annotations)[fields + tuple(annotations)]

    def groupby(self, *fields, **annotations) -> 'QuerySet':
        """Return a grouped `QuerySet`_.

        The queryset is iterable in the same manner as ``itertools.groupby``.
        Additionally the :meth:`reduce` functions will return annotated querysets.
        """
        qs = self.annotate(**annotations)
        qs._groupby = fields + tuple(annotations)
        return qs

    def annotate(self, *args, **kwargs) -> 'QuerySet':
        """Annotate extended to also handle mapping values, as a `Case`_ expression.

        :param kwargs: ``field={Q_obj: value, ...}, ...``

        As a provisional feature, an optional ``default`` key may be specified.
        """
        for field, value in kwargs.items():
            if Case.isa(value):
                kwargs[field] = Case.defaultdict(value)
        return super().annotate(*args, **kwargs)

    def value_counts(self, alias: str = 'count') -> 'QuerySet':
        """Return annotated value counts."""
        return self.items(*self._fields, **{alias: F.count()})

    def sort_values(self, reverse=False) -> 'QuerySet':
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

    def update(self, **kwargs) -> int:
        """Update extended to also handle mapping values, as a `Case`_ expression.

        :param kwargs: ``field={Q_obj: value, ...}, ...``
        """
        for field, value in kwargs.items():
            if Case.isa(value):
                kwargs[field] = Case(value, default=F(field))
        return super().update(**kwargs)

    def change(self, defaults: Mapping = {}, **kwargs) -> int:
        """Update and return number of rows that actually changed.

        For triggering on-change logic without fetching first.

        ``if qs.change(status=...):`` status actually changed

        ``qs.change({'last_modified': now}, status=...)`` last_modified only updated if status updated

        :param defaults: optional mapping which will be updated conditionally, as with ``update_or_create``.
        """
        return self.exclude(**kwargs).update(**dict(defaults, **kwargs))

    def changed(self, **kwargs) -> dict:
        """Return first mapping of fields and values which differ in the db.

        Also efficient enough to be used in boolean contexts, instead of ``exists``.
        """
        row = self.exclude(**kwargs).values(*kwargs).first() or {}
        return {field: value for field, value in row.items() if value != kwargs[field]}

    def exists(self, count: int = 1) -> bool:
        """Return whether there are at least the specified number of rows."""
        if count == 1:
            return super().exists()
        return (self[:count].count() if self._result_cache is None else len(self)) >= count


@models.Field.register_lookup
class NotEqual(models.Lookup):
    """Missing != operator."""

    lookup_name = 'ne'

    def as_sql(self, *args):
        lhs, lhs_params = self.process_lhs(*args)
        rhs, rhs_params = self.process_rhs(*args)
        return f'{lhs} <> {rhs}', (lhs_params + rhs_params)


class Query(models.sql.Query):
    """Allow __ne=None lookup."""

    def build_lookup(self, lookups, lhs, rhs):
        if rhs is None and lookups[-1:] == ['ne']:
            rhs, lookups[-1] = False, 'isnull'
        return super().build_lookup(lookups, lhs, rhs)


class Manager(models.Manager):
    def get_queryset(self):
        return QuerySet(self.model, Query(self.model), self._db, self._hints)

    def __getitem__(self, pk) -> QuerySet:
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

    def upsert(self, defaults: Mapping = {}, **kwargs) -> Union[int, models.Model]:
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

    def bulk_changed(self, field, data: Mapping, key: str = 'pk') -> dict:
        """Return mapping of values which differ in the db.

        :param field: value column
        :param data: ``{pk: value, ...}``
        :param key: unique key column
        """
        rows = self.filter(F(key).isin(data))[key, field].iterator()
        return {pk: value for pk, value in rows if value != data[pk]}

    def bulk_change(self, field, data: Mapping, key: str = 'pk', conditional=False, **kwargs) -> int:
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
        updates = collections.defaultdict(list)  # type: dict
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
        super().__init__(*cases, default=Value(default), **extra)

    @classmethod
    def defaultdict(cls, conds):
        conds = dict(conds)
        return cls(conds, default=conds.pop('default', None))

    @classmethod
    def isa(cls, value):
        return isinstance(value, Mapping) and any(isinstance(key, models.Q) for key in value)


def EnumField(enum, display: Callable = None, **options) -> models.Field:
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
