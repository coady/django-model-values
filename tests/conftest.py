import django
from django.conf import settings


def pytest_report_header(config):
    return 'Django ' + django.__version__


pytest_plugins = ('django',)
settings.configure(INSTALLED_APPS=['tests'], DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3'}})
