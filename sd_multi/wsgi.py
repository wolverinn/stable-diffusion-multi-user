"""
WSGI config for tti project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

from os.path import join,dirname,abspath
PROJECT_DIR = dirname(dirname(abspath(__file__)))
import sys
sys.path.insert(0,PROJECT_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sd_multi.settings')

application = get_wsgi_application()
