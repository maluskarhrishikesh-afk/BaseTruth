import starlette.routing as m
import inspect
src = inspect.getsource(m)
idx = src.find('async def app(self')
src2 = src[idx:]
idx2 = src2.find('websocket')
while idx2 >= 0:
    snippet = src2[max(0,idx2-50):idx2+200]
    if 'close' in snippet or 'Not Found' in snippet or '404' in snippet or '403' in snippet:
        print('--- AT ' + str(idx2) + ' ---')
        print(snippet)
    idx2 = src2.find('websocket', idx2+1)
