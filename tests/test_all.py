import math
import django
from django.db import models
from django.db.models import functions
from django.utils import timezone
import pytest
from .models import Book
from model_values import Case, EnumField, F, gis, transform

pytestmark = pytest.mark.django_db


@pytest.fixture
def books():
    for quantity in (10, 10):
        Book.objects.create(author='A', quantity=quantity)
    for quantity in (2, 1, 2):
        Book.objects.create(author='B', quantity=quantity)
    return Book.objects.all()


def test_queryset(books):
    assert books.filter(id__ne=None).exists(5)
    assert books and books.exists() and not books.exists(6)
    assert set(books['author']) == set(books[F.author]) == {'A', 'B'}
    assert dict(books[F.id, 'author']) == {1: 'A', 2: 'A', 3: 'B', 4: 'B', 5: 'B'}
    assert set(books[F.author.lower()]) == {'a', 'b'}
    assert dict(books['id', F.author.lower()]) == {1: 'a', 2: 'a', 3: 'b', 4: 'b', 5: 'b'}

    assert len(books['quantity'] < 2) == 1
    assert len(books['quantity'] <= 2) == 3
    assert len(books['quantity'] > 2) == 2
    assert len(books['quantity'] >= 2) == 4
    assert len(books['quantity'] == 2) == 2
    assert len(books['quantity'] != 2) == 3

    quant = books['quantity']
    assert 10 in quant and quant._result_cache is None
    assert quant and 10 in quant
    assert books[0] in books.all()
    assert ('A', 10) in books['author', 'quantity']
    assert list(quant.sort_values()) == [1, 2, 2, 10, 10]
    assert list(quant.sort_values(reverse=True)) == [10, 10, 2, 2, 1]

    now = timezone.now()
    assert books.filter(author='B').change({'last_modified': now}, quantity=2) == 1
    assert len(books['last_modified'] == now) == 1
    books['quantity'] = {F.author == 'B': 3}
    assert set(books['quantity']) == {3, 10}
    assert Book.objects.upsert({'quantity': 0}, pk=1) == 1
    assert Book.objects.upsert(pk=0) == 0  # simulates race condition
    book = Book.objects.upsert({'quantity': F.quantity + 1}, pk=0)
    assert book.pk == 0 and book.quantity == 1
    with pytest.raises(TypeError):
        books['quantity'] = {}


def test_manager(books):
    assert 1 in Book.objects
    assert Book.objects[1]['id'].first() == 1
    assert Book.objects.bulk_changed('quantity', {3: 2, 4: 2, 5: 2}) == {4: 1}
    assert Book.objects.bulk_changed('quantity', {'A': 5}, key='author') == {'A': 10}
    now = timezone.now()
    assert Book.objects.bulk_change('quantity', {3: 2, 4: 2}, last_modified=now) == 1
    timestamps = dict(books.filter(quantity=2)['id', 'last_modified'])
    assert len(timestamps) == 3 and timestamps[3] < timestamps[5] < timestamps[4] == now
    assert Book.objects.bulk_change('quantity', {3: 2, 4: 3}, key='id', conditional=True) == 1
    assert set(books.filter(quantity=2)['id']) == {3, 5}
    assert Book.objects[1].changed(quantity=5) == {'quantity': 10}
    del Book.objects[1]
    assert 1 not in Book.objects


def test_aggregation(books):
    assert set(books['author'].annotate(models.Max('quantity'))) == {'A', 'B'}
    assert dict(books['author'].value_counts()) == {'A': 2, 'B': 3}

    assert books['author', 'quantity'].reduce(models.Max, models.Min) == ('B', 1)
    assert books['author', 'quantity'].min() == ('A', 1)
    assert books['quantity'].min() == 1
    assert books['quantity'].max() == 10
    assert books['quantity'].sum() == 25
    assert books['quantity'].mean() == 5.0

    groups = books['quantity'].groupby('author')
    assert {key: sorted(values) for key, values in groups} == {'A': [10, 10], 'B': [1, 2, 2]}
    assert dict(groups.min()) == {'A': 10, 'B': 1}
    assert dict(groups.max()) == {'A': 10, 'B': 2}
    assert dict(groups.sum()) == {'A': 20, 'B': 5}
    assert dict(groups.mean()) == {'A': 10, 'B': 5.0 / 3}
    assert isinstance(groups.var(), models.QuerySet)
    assert isinstance(groups.std(), models.QuerySet)
    key, values = next(iter(books.values('title', 'last_modified').groupby('author', 'quantity')))
    assert key == ('A', 10) and next(values)[0] == ''

    groups = books['quantity'].groupby(author=F.author.lower())
    assert dict(groups.sum()) == {'a': 20, 'b': 5}
    counts = books[F.author.lower()].value_counts()
    assert dict(counts) == {'a': 2, 'b': 3}
    assert dict(counts[F('count') > 2]) == {'b': 3}
    amounts = books[{F.quantity <= 1: 'low', F.quantity >= 10: 'high'}]
    assert dict(amounts.value_counts()) == {'low': 1, None: 2, 'high': 2}
    groups = books.groupby(amount={F.quantity <= 1: 'low', F.quantity >= 10: 'high', 'default': 'medium'})
    with pytest.raises(Exception):
        Case({models.Q(): None}).output_field

    expr = books.values_list(F.quantity * -1)
    assert type(expr.sum()) is tuple
    key, values = next(iter(expr.groupby('author')))
    assert set(map(type, values)) == {tuple}


def test_functions(books):
    book = Book.objects[1]
    assert book['quantity'].first() == 10
    book['quantity'] += 1
    assert book['quantity'].first() == 11
    book['quantity'] -= 1
    assert book['quantity'].first() == 10
    book['quantity'] *= 2
    assert book['quantity'].first() == 20
    book['quantity'] /= 2
    assert book['quantity'].first() == 10
    book['quantity'] %= 7
    assert book['quantity'].first() == 3
    book['quantity'] **= 2
    assert book['quantity'].first() == 9

    assert (F.author != None) == models.Q(author__isnull=False)  # noqa: E711
    assert isinstance(F.coalesce('author', 'title'), functions.Coalesce)
    assert isinstance(F.author.concat('title'), functions.Concat)
    assert isinstance(F.author.length(), functions.Length)
    assert isinstance(F.title.lower(), functions.Lower)
    assert isinstance(F.title.upper(), functions.Upper)
    assert isinstance(F.title[:10], functions.Substr)
    with pytest.raises(AssertionError):
        F.title[:-10]
    with pytest.raises(AttributeError):
        F.name
    with pytest.raises(TypeError):
        iter(F.title)
    assert hash(F.title)
    assert not (F.author == models.F('title'))
    ((field, values),) = transform('op', F.author.coalesce('title'), None).children
    assert field == 'author__op' and values == (F.title, None)

    assert isinstance(F.title.greatest('author'), functions.Greatest)
    assert isinstance(F.title.least('author'), functions.Least)
    assert F.now is functions.Now
    assert isinstance(F.quantity.cast(models.FloatField()), functions.Cast)
    assert isinstance(F.last_modified.extract('year'), functions.Extract)
    assert isinstance(F.last_modified.trunc('year'), functions.Trunc)

    for name, func in F.lookups.items():
        models.CharField.register_lookup(func, name)
    assert books[F.author.length <= 1]
    assert books[F.author.lower == 'a']
    assert books[F.author.upper == 'A']


def test_2(books):
    row = books['id', 'author'].first()
    assert (row.id, row.author) == row
    row = books['author',].min()
    assert (row.author__min,) == row
    key, values = next(iter(books['quantity',].groupby('author')))
    assert next(values).quantity
    assert dict(books[F.author.find('A')].value_counts()) == {-1: 3, 0: 2}

    assert isinstance(F.quantity.cume_dist(), functions.CumeDist)
    assert isinstance(F.quantity.dense_rank(), functions.DenseRank)
    assert isinstance(F.quantity.first_value(), functions.FirstValue)
    assert isinstance(F.quantity.lag(), functions.Lag)
    assert isinstance(F.quantity.last_value(), functions.LastValue)
    assert isinstance(F.quantity.lead(), functions.Lead)
    assert isinstance(F.quantity.nth_value(), functions.NthValue)
    assert F.ntile is functions.Ntile
    assert isinstance(F.quantity.percent_rank(), functions.PercentRank)
    assert isinstance(F.quantity.rank(), functions.Rank)
    assert isinstance(F.quantity.row_number(), functions.RowNumber)

    point = 'POINT(0 0)'
    if gis:
        assert isinstance(F.location.azimuth(point), gis.functions.Azimuth)
        assert isinstance(F.location.line_locate_point(point), gis.functions.LineLocatePoint)


def test_2_1():
    assert (F.quantity.chr == '').children == [('quantity__chr', '')]
    assert isinstance(F.quantity.chr(), functions.Chr)
    assert (F.author.ord == 0).children == [('author__ord', 0)]
    assert isinstance(F.author.ord(), functions.Ord)

    assert isinstance(F.title[-10:], functions.Right)
    assert isinstance(F.author.replace('A', 'B'), functions.Replace)
    assert isinstance(F.author.repeat(3), functions.Repeat)

    assert isinstance(F.title.strip(), functions.Trim)
    assert isinstance(F.title.lstrip(), functions.LTrim)
    assert isinstance(F.title.rstrip(), functions.RTrim)
    assert isinstance(F.author.ljust(1), functions.LPad)
    assert isinstance(F.author.rjust(1), functions.RPad)

    if gis:
        assert isinstance(F.location.force_polygon_cw(), gis.functions.ForcePolygonCW)


def test_2_2():
    assert isinstance(F.x.nullif('y'), functions.NullIf)
    assert isinstance(reversed(F.x), functions.Reverse)
    assert isinstance(abs(F.x), functions.Abs)
    assert isinstance(F.x.acos(), functions.ACos)
    assert isinstance(F.x.asin(), functions.ASin)
    assert isinstance(F.x.atan(), functions.ATan)
    assert isinstance(F.x.atan2('y'), functions.ATan2)
    assert isinstance(math.ceil(F.x), functions.Ceil)
    assert isinstance(F.x.cos(), functions.Cos)
    assert isinstance(F.x.cot(), functions.Cot)
    assert isinstance(F.x.degrees(), functions.Degrees)
    assert isinstance(F.x.exp(), functions.Exp)
    assert isinstance(math.floor(F.x), functions.Floor)
    assert isinstance(F.x.log(), functions.Log)
    assert isinstance(F.x.log(2), functions.Log)
    assert isinstance(F.x % 2, functions.Mod)
    assert isinstance(2 % F.x, functions.Mod)
    assert isinstance(F.pi, functions.Pi)
    assert isinstance(F.x ** 2, functions.Power)
    assert isinstance(2 ** F.x, functions.Power)
    assert isinstance(F.x.radians(), functions.Radians)
    assert isinstance(round(F.x), functions.Round)
    assert isinstance(F.x.sin(), functions.Sin)
    assert isinstance(F.x.sqrt(), functions.Sqrt)
    assert isinstance(F.x.tan(), functions.Tan)

    assert isinstance(F.x.acos, F)
    assert isinstance(F.x.asin, F)
    assert isinstance(F.x.atan, F)
    assert isinstance(F.x.atan2, F)
    assert isinstance(F.x.cos, F)
    assert isinstance(F.x.cot, F)
    assert isinstance(F.x.degrees, F)
    assert isinstance(F.x.exp, F)
    assert isinstance(F.x.radians, F)
    assert isinstance(F.x.sin, F)
    assert isinstance(F.x.sqrt, F)
    assert isinstance(F.x.tan, F)


@pytest.mark.skipif(django.VERSION < (3,), reason='requires django >=3')
def test_3():
    assert isinstance(F.x.sign(), functions.Sign)
    assert isinstance(F.x.md5(), functions.MD5)
    assert isinstance(F.x.sha1(), functions.SHA1)
    assert isinstance(F.x.sha224(), functions.SHA224)
    assert isinstance(F.x.sha256(), functions.SHA256)
    assert isinstance(F.x.sha384(), functions.SHA384)
    assert isinstance(F.x.sha512(), functions.SHA512)

    assert isinstance(F.x.sign, F)
    assert isinstance(F.x.md5, F)


def test_lookups(books):
    assert books[F.last_modified.year == timezone.now().year].count() == 5
    assert isinstance(F.quantity.min(), models.Min)
    assert isinstance(F.quantity.max(), models.Max)
    assert isinstance(F.quantity.sum(), models.Sum)
    assert isinstance(F.quantity.mean(), models.Avg)
    assert str(F.quantity.count()).startswith('Count(F(quantity)')
    assert str(F.count(distinct=True)) == "Count('*', distinct=True)"
    assert isinstance(F.quantity.var(sample=True), models.Variance)
    assert isinstance(F.quantity.std(sample=True), models.StdDev)
    exprs = list(map(F.author.contains, 'AB'))
    assert str(F.any(exprs)) == "(OR: ('author__contains', 'A'), ('author__contains', 'B'))"
    assert str(F.all(exprs)) == "(AND: ('author__contains', 'A'), ('author__contains', 'B'))"

    authors = books['author']
    assert set(authors.isin('AB')) == {'A', 'B'}
    assert set(authors.iexact('a')) == {'A'}
    assert set(authors.contains('A')) == {'A'}
    assert set(authors.icontains('a')) == {'A'}
    assert set(authors.startswith('A')) == {'A'}
    assert set(authors.istartswith('a')) == {'A'}
    assert set(authors.endswith('A')) == {'A'}
    assert set(authors.iendswith('a')) == {'A'}
    assert set(authors.range('A', 'B')) == {'A', 'B'}
    assert set(authors.regex('A')) == {'A'}
    assert set(authors.iregex('a')) == {'A'}


def test_model(books):
    book = Book.objects.get(pk=1)
    assert list(book.object) == [book]
    assert len(Book.in_stock) == 5
    assert book.changed(quantity=5) == {'quantity': 10}
    assert book.changed(quantity=10) == {}
    assert book.update(quantity=2) == 1
    assert book.quantity == 2 and 2 in book.object['quantity']


def test_spatial_lookups():
    point = 'POINT(0 0)'
    assert F.location.is_valid.children == [('location__isvalid', True)]
    assert F.location.contains(point, bb=True).children == [('location__bbcontains', point)]
    assert F.location.contains(point, properly=True).children == [('location__contains_properly', point)]

    assert F.location.overlaps(point).children == [('location__overlaps', point)]
    assert F.location.overlaps(point, bb=True).children == [('location__bboverlaps', point)]
    assert F.location.overlaps(point, 'left').children == [('location__overlaps_left', point)]
    assert F.location.overlaps(point, 'right').children == [('location__overlaps_right', point)]
    assert F.location.overlaps(point, 'above').children == [('location__overlaps_above', point)]
    assert F.location.overlaps(point, 'below').children == [('location__overlaps_below', point)]

    assert F.location.within(point).children == [('location__within', point)]
    assert F.location.within(point, 0).children == [('location__dwithin', (point, 0))]

    assert F.location.contained(point).children == [('location__contained', point)]
    assert F.location.coveredby(point).children == [('location__coveredby', point)]
    assert F.location.covers(point).children == [('location__covers', point)]
    assert F.location.crosses(point).children == [('location__crosses', point)]
    assert F.location.disjoint(point).children == [('location__disjoint', point)]
    assert F.location.equals(point).children == [('location__equals', point)]
    assert F.location.intersects(point).children == [('location__intersects', point)]
    assert F.location.relate(point, '').children == [('location__relate', (point, ''))]
    assert F.location.touches(point).children == [('location__touches', point)]

    assert (F.location << point).children == F.location.left(point).children == [('location__left', point)]
    assert (F.location >> point).children == F.location.right(point).children == [('location__right', point)]
    assert F.location.above(point).children == [('location__strictly_above', point)]
    assert F.location.below(point).children == [('location__strictly_below', point)]


@pytest.mark.skipif(not gis, reason='requires spatial lib')
def test_spatial_functions(books):
    from django.contrib.gis.geos import Point

    point = Point(0, 0, srid=4326)

    assert isinstance(F.location.area, gis.functions.Area)
    assert isinstance(F.location.geojson(), gis.functions.AsGeoJSON)
    assert isinstance(F.location.gml(), gis.functions.AsGML)
    assert isinstance(F.location.kml(), gis.functions.AsKML)
    assert isinstance(F.location.svg(), gis.functions.AsSVG)
    assert isinstance(F.location.bounding_circle(), gis.functions.BoundingCircle)
    assert isinstance(F.location.centroid, gis.functions.Centroid)
    assert isinstance(F.location.envelope, gis.functions.Envelope)
    assert isinstance(F.location.geohash(), gis.functions.GeoHash)
    assert isinstance(F.location.make_valid(), gis.functions.MakeValid)
    assert isinstance(F.location.mem_size, gis.functions.MemSize)
    assert isinstance(F.location.num_geometries, gis.functions.NumGeometries)
    assert isinstance(F.location.num_points, gis.functions.NumPoints)
    assert isinstance(F.location.perimeter, gis.functions.Perimeter)
    assert isinstance(F.location.point_on_surface, gis.functions.PointOnSurface)
    assert isinstance(F.location.reverse(), gis.functions.Reverse)
    assert isinstance(F.location.scale(0, 0), gis.functions.Scale)
    assert isinstance(F.location.snap_to_grid(0), gis.functions.SnapToGrid)
    assert isinstance(F.location.transform(point.srid), gis.functions.Transform)
    assert isinstance(F.location.translate(0, 0), gis.functions.Translate)

    dist = F.location.distance(point)
    assert isinstance(dist, gis.functions.Distance)
    fields, items = zip(*F.all([(dist < 0), (dist <= 0), (dist > 0), (dist >= 0), dist.within(0)]).children)
    assert fields == (
        'location__distance_lt',
        'location__distance_lte',
        'location__distance_gt',
        'location__distance_gte',
        'location__dwithin',
    )
    assert items == ((point, 0),) * 5
    ((field, values),) = (F.location.distance(point) > 0).children
    assert values == (point, 0)

    assert isinstance(F.location.difference(point), gis.functions.Difference)
    assert isinstance(F.location.intersection(point), gis.functions.Intersection)
    assert isinstance(F.location.symmetric_difference(point), gis.functions.SymDifference)
    assert isinstance(F.location.union(point), gis.functions.Union)

    assert type(books).collect.__name__ == 'Collect'
    assert type(books).extent.__name__ == 'Extent'
    assert type(books).extent3d.__name__ == 'Extent3D'
    assert type(books).make_line.__name__ == 'MakeLine'
    assert type(books).union.__name__ == 'Union'
    with pytest.raises(ValueError, match="Geospatial aggregates only allowed on geometry fields."):
        books['id'].collect()


def test_enum():
    enum = pytest.importorskip('enum')

    @EnumField
    class gender(enum.Enum):
        M = 'Male'
        F = 'Female'

    assert gender.max_length == 1
    assert dict(gender.choices) == {'M': 'Male', 'F': 'Female'}

    class Gender(enum.Enum):
        MALE = 0
        FEMALE = 1

    gender = EnumField(Gender, str.title)
    assert isinstance(gender, models.IntegerField)
    assert dict(gender.choices) == {0: 'Male', 1: 'Female'}
