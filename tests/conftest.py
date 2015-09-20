import django
from django.conf import settings


def pytest_report_header(config):
    return 'Django ' + django.get_version(django.VERSION)

settings.configure(
    INSTALLED_APPS=['tests'],
    DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3'}},
    SILENCED_SYSTEM_CHECKS=['1_7.W001'],
)
