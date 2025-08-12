# probe_s3.py
import requests
URL = "https://stat468-final-project.s3.us-east-1.amazonaws.com/models/mp_value_ridge_pipeline.joblib"
r = requests.get(URL, timeout=60)
print("status:", r.status_code)
print("content-type:", r.headers.get("Content-Type"))
print("bytes:", len(r.content))
print("start:", r.content[:120])
