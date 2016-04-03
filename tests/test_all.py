from django.db import models
from django.db.models import functions
from django.utils import timezone
from django_dynamic_fixture import G
import pytest
from .models import Book
from model_values import F

pytestmark = pytest.mark.django_db


@pytest.fixture
def books():
    for quantity in (10, 10):
        G(Book, author='A', quantity=quantity)
    for quantity in (2, 1, 2):
        G(Book, author='B', quantity=quantity)
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
    assert all(value.title and value.last_modified for value in values)

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

    assert isinstance(F.author | 'title', functions.Coalesce)
    assert isinstance(F.author.concat('title'), functions.Concat)
    assert isinstance(F.author.length(), functions.Length)
    assert isinstance(F.title.lower(), functions.Lower)
    assert isinstance(F.title.upper(), functions.Upper)
    assert isinstance(F.title[:10], functions.Substr)
    with pytest.raises(AssertionError):
        F.title[:-10]
    if hasattr(type(F), 'now'):
        assert isinstance(F.title.greatest('author'), functions.Greatest)
        assert isinstance(F.title.least('author'), functions.Least)
        assert F.now is functions.Now


def test_lookups(books):
    assert books[F.last_modified.year == timezone.now().year].count() == 5
    assert str(F.author.search('')) == "(AND: ('author__search', ''))"
    assert isinstance(F.quantity.min(), models.Min)
    assert isinstance(F.quantity.max(), models.Max)
    assert isinstance(F.quantity.sum(), models.Sum)
    assert isinstance(F.quantity.mean(), models.Avg)
    assert isinstance(F.count(distinct=True), models.Count)
    assert isinstance(F.quantity.var(sample=True), models.Variance)
    assert isinstance(F.quantity.std(sample=True), models.StdDev)
    ordering = -F.user.created
    assert ordering.expression.name == 'user__created' and ordering.descending

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
