# -*- coding: utf-8 -*-
import shutil

SRC = 'app2.py'
shutil.copy(SRC, 'app2_backup2.py')
print('Backup created: app2_backup2.py')

with open(SRC, encoding='utf-8') as f:
    c = f.read()

log = []

# ===== 1) Rate limiter + guvenlik basliklari + temiz hatalar =====
anchor = 'app = Flask(__name__)'
if anchor not in c:
    log.append('MISS  Flask app satiri')
else:
    new = anchor + '''

# --- Security: rate limiting ---
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(get_remote_address, app=app,
                      default_limits=["120 per minute"],
                      storage_uri="memory://")
except ImportError:
    class _NoLimiter:
        def limit(self, *_a, **_k):
            def deco(f): return f
            return deco
    limiter = _NoLimiter()

# --- Security: response headers ---
@app.after_request
def _secure_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp

# --- Security: clean error responses (stack trace sizdirma) ---
@app.errorhandler(404)
def _e404(e): return jsonify({"error": "not found"}), 404

@app.errorhandler(429)
def _e429(e): return jsonify({"error": "rate limit exceeded, slow down"}), 429

@app.errorhandler(500)
def _e500(e): return jsonify({"error": "server error"}), 500
'''
    c = c.replace(anchor, new, 1)
    log.append('OK    limiter + headers + error handlers')

# ===== 2) /analyze icin siki limit =====
old = '@app.route("/analyze", methods=["POST"])'
if old not in c:
    log.append('MISS  /analyze route')
else:
    c = c.replace(old, old + '\n@limiter.limit("20 per minute")', 1)
    log.append('OK    /analyze rate limit 20/dk')

# ===== 3) Girdi dogrulama =====
old = 'd = request.json'
if old not in c:
    log.append('MISS  request.json')
else:
    new = '''d = request.get_json(silent=True)
    if not isinstance(d, dict):
        return jsonify({"error": "invalid request body"}), 400
    if len(str(d.get("vessel", ""))) > 60 or len(str(d.get("route", ""))) > 60:
        return jsonify({"error": "invalid parameters"}), 400
    try:
        d = dict(d)
        d["speed"] = float(d.get("speed", 12))
        d["draft"] = float(d.get("draft", 8.5))
        d["swh"]   = float(d.get("swh", 1.2))
        d["sst"]   = float(d.get("sst", 22))
        d["days"]  = int(d.get("days", 280))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid parameters"}), 400
    if not (1 <= d["speed"] <= 30 and 1 <= d["draft"] <= 26
            and 0 <= d["swh"] <= 20 and -2 <= d["sst"] <= 40
            and 1 <= d["days"] <= 366):
        return jsonify({"error": "parameters out of range"}), 400'''
    c = c.replace(old, new, 1)
    log.append('OK    input validation (speed/draft/swh/sst/days sinirlari)')

print('\n'.join(log))
if any(l.startswith('MISS') for l in log):
    print('\n!!! MISS var - dosya YAZILMADI.')
else:
    with open(SRC, 'w', encoding='utf-8') as f:
        f.write(c)
    print('\napp2.py updated!') 