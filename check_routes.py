from basetruth.api import app
for r in app.routes:
    t = type(r).__name__
    p = getattr(r, 'path', '')
    print(t + ': ' + p)
