import fastapi.exceptions as fe
import inspect, starlette.websockets as sw
print('WebSocketRequestValidationError bases:', fe.WebSocketRequestValidationError.__mro__)
print('WebSocketException:', sw.WebSocketException.__mro__)
# check if there's a handler for WebSocketRequestValidationError
from basetruth.api import app
print('Exception handlers:')
for k, v in app.exception_handlers.items():
    print(k, '->', v)
