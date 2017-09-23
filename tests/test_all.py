import django
from django.db import models
from django.db.models import functions
from django.utils import timezone
import pytest
from .models import Book
from model_values import F, gis

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
    assert set(books['author']) == set(books[F.author]) == {'A', 'B'}
    assert dict(books[F.id, 'author']) == {1: 'A', 2: 'A', 3: 'B', 4: 'B', 5: 'B'}

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

    now = timezone.now()
    assert books.filter(author='B').modify({'last_modified': now}, quantity=2) == 1
    assert len(books['last_modified'] == now) == 1
    assert books.filter(author='B').modify({'last_modified': timezone.now()}, quantity=2) == 0
    assert len(books['last_modified'] == now) == 1
    books['quantity'] = {F.author == 'B': 3}
    assert set(books['quantity']) == {3, 10}
    assert books.upsert({'quantity': 0}, pk=1) == 1
    assert books.upsert(pk=0) == 0  # simulates race condition
    book = books.upsert({'quantity': F.quantity + 1}, pk=0)
    assert book.pk == 0 and book.quantity == 1
    assert books.upsert({'quantity': F.quantity + 1}, pk=0) == 1
    assert books['quantity'].get(pk=0) == 2


def test_manager(books):
    assert 1 in Book.objects
    assert Book.objects[1]['id'].first() == 1
    assert Book.objects.bulk_changed('quantity', {3: 2, 4: 2, 5: 2}) == {4: 1}
    now = timezone.now()
    assert Book.objects.bulk_update('quantity', {3: 2, 4: 2}, changed=True, last_modified=now) == 1
    timestamps = dict(books.filter(quantity=2)['id', 'last_modified'])
    assert len(timestamps) == 3 and timestamps[3] < timestamps[5] < timestamps[4] == now
    assert Book.objects.bulk_update('quantity', {3: 2, 4: 3}, conditional=True) == 2
    assert set(books.filter(quantity=2)['id']) == {3, 5}
    assert Book.objects.changed(1, quantity=5) == {'quantity': 10}
    assert Book.objects.changed(1, quantity=10) == {}
    del Book.objects[1]
    assert 1 not in Book.objects


def test_aggregation(books):
    assert books.values('author').annotate(models.Max('quantity'))
    assert set(books['author', ].annotate()) == {('A',), ('B',)}
    assert dict(books['author'].annotate(models.Max('quantity'))) == {'A': 10, 'B': 2}
    assert dict(books['author'].value_counts()) == {'A': 2, 'B': 3}

    values = books['author', 'quantity'].reduce(models.Max, models.Min)
    assert values.author__max == 'B' and values.quantity__min == 1 and values == ('B', 1)
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
    assert key == ('A', 10)
    assert sum((value[0] == value.title) and bool(value.last_modified) for value in values) == 2

    groups = books['quantity'].groupby(author=F.author.lower())
    assert dict(groups.sum()) == {'a': 20, 'b': 5}
    counts = books.groupby(alias=F.author.lower()).value_counts()
    assert dict(counts) == {'a': 2, 'b': 3}
    assert dict(counts[F('count') > 2]) == {'b': 3}
    groups = books.groupby(amount={F.quantity < 10: 'low', F.quantity >= 10: 'high'})
    assert dict(groups.value_counts()) == {'low': 3, 'high': 2}


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

    assert isinstance(F.coalesce('author', 'title'), functions.Coalesce)
    assert isinstance(F.author.concat('title'), functions.Concat)
    assert isinstance(F.author.length(), functions.Length)
    assert isinstance(F.title.lower(), functions.Lower)
    assert isinstance(F.title.upper(), functions.Upper)
    assert isinstance(F.title[:10], functions.Substr)
    with pytest.raises(AssertionError):
        F.title[:-10]


@pytest.mark.skipif(django.VERSION < (1, 10), reason='requires django 1.10+')
def test_new_functions():
    assert isinstance(F.title.greatest('author'), functions.Greatest)
    assert isinstance(F.title.least('author'), functions.Least)
    assert F.now is functions.Now
    assert isinstance(F.quantity.cast(models.FloatField()), functions.Cast)
    assert isinstance(F.last_modified.extract('year'), functions.Extract)
    assert isinstance(F.last_modified.trunc('year'), functions.Trunc)


def test_lookups(books):
    assert books[F.last_modified.year == timezone.now().year].count() == 5
    assert str(F.author.search('')) == "(AND: ('author__search', ''))"
    assert isinstance(F.quantity.min(), models.Min)
    assert isinstance(F.quantity.max(), models.Max)
    assert isinstance(F.quantity.sum(), models.Sum)
    assert isinstance(F.quantity.mean(), models.Avg)
    assert str(F.quantity.count()) == "Count(F(quantity), distinct=False)"
    assert str(F.count(distinct=True)) == "Count('*', distinct=True)"
    assert isinstance(F.quantity.var(sample=True), models.Variance)
    assert isinstance(F.quantity.std(sample=True), models.StdDev)
    ordering = -F.user.created
    assert ordering.expression.name == 'user__created' and ordering.descending
    ordering = +F.user.created
    assert ordering.expression.name == 'user__created' and not ordering.descending
    exprs = list(map(F.author.contains, 'AB'))
    assert str(F.any(exprs)) == "(OR: ('author__contains', 'A'), ('author__contains', 'B'))"
    assert str(F.all(exprs)) == "(AND: ('author__contains', 'A'), ('author__contains', 'B'))"

    authors = books['author']
    assert set(authors.in_('A', 'B')) == {'A', 'B'}
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
    assert isinstance(F.location.distance(point), gis.functions.Distance)
    assert isinstance(F.location.envelope, gis.functions.Envelope)
    assert isinstance(F.location.force_rhr(), gis.functions.ForceRHR)
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
