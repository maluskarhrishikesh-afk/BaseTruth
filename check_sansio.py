import uvicorn.protocols.websockets.websockets_sansio_impl as m
import inspect
src = inspect.getsource(m)
print(src)
