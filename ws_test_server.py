import asyncio, uvicorn
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket('/ws/test')
async def ws_test(ws: WebSocket):
    await ws.accept()
    await ws.send_text('hello')
    await ws.close()

config = uvicorn.Config(app, host='127.0.0.1', port=8001, ws='websockets-sansio', log_level='info')
server = uvicorn.Server(config)
asyncio.run(server.serve())
