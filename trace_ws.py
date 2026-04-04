import asyncio, uvicorn, sys
from fastapi import FastAPI, WebSocket
from fastapi.exceptions import WebSocketRequestValidationError
from starlette.middleware.cors import CORSMiddleware

from basetruth.api import create_app

# Get the actual app
app = create_app()

# Override the WebSocket validation exception handler to log errors
from fastapi.exception_handlers import websocket_request_validation_exception_handler as _orig

async def patched_ws_handler(websocket: WebSocket, exc: WebSocketRequestValidationError):
    print('WS VALIDATION ERROR: ' + repr(exc.errors()), flush=True)
    return await _orig(websocket, exc)

app.add_exception_handler(WebSocketRequestValidationError, patched_ws_handler)

# Wrap the app to trace websocket messages
original_app = app.__call__

async def traced_app(scope, receive, send):
    if scope.get('type') == 'websocket':
        async def traced_send(msg):
            t = msg.get('type'); p = scope.get('path')
            print('SEND: ' + str(t) + ' path=' + str(p), flush=True)
            return await send(msg)
        async def traced_receive():
            msg = await receive()
            t = msg.get('type')
            print('RECV: ' + str(t), flush=True)
            return msg
        return await original_app(scope, traced_receive, traced_send)
    return await original_app(scope, receive, send)

config = uvicorn.Config(traced_app, host='127.0.0.1', port=8002, ws='websockets-sansio', log_level='warning')
server = uvicorn.Server(config)
asyncio.run(server.serve())
