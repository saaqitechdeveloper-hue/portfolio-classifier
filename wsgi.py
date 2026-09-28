import sys
import os

path = '/home/saaqitech/portfolio-classifier'
if path not in sys.path:
    sys.path.append(path)

from api.app_flask import app as application
