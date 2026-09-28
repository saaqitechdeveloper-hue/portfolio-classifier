from a2wsgi import ASGIMiddleware
from api.app_onnx import app

application = ASGIMiddleware(app)
