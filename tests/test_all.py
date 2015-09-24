from django.db import models
from django.utils import timezone
from django_dynamic_fixture import G
import pytest
from .models import Book

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
    assert set(books['author']) == {'A', 'B'}
    assert dict(books['id', 'author']) == {1: 'A', 2: 'A', 3: 'B', 4: 'B', 5: 'B'}

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
    assert books.remove() == 5
    assert not books


def test_manager(books):
    assert 1 in Book.objects
    assert Book.objects[1]['id'].first() == 1
    assert Book.objects.update_rows(dict.fromkeys([3, 4, 5], {'quantity': 2})) == {4}
    assert 4 in (books['quantity'] == 2)['id']
    assert Book.objects.update_columns('quantity', dict.fromkeys([3, 4, 5], 1)) == {1: 3}
    assert len(books['quantity'] == 1) == 3
    assert Book.objects.changed(1, quantity=5) == {'quantity': 10}
    assert Book.objects.changed(1, quantity=10) == {}


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
    key, values = next(iter(books.values('title', 'last_modified').groupby('author', 'quantity')))
    assert key == ('A', 10)
    assert all(value.title and value.last_modified for value in values)


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


def test_model(books):
    book = Book.objects.get(pk=1)
    assert list(book.object) == [book]
    assert len(Book.in_stock) == 5
    assert book.changed(quantity=5) == {'quantity': 10}
    assert book.changed(quantity=10) == {}
    assert book.update(quantity=2) == 1
    assert book.quantity == 2 and 2 in book.object['quantity']
