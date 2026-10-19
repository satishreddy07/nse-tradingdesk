import subprocess

res = subprocess.run(["ssh", "root@104.168.91.147", "ps aux | grep uvicorn"], capture_output=True, text=True)
print("Processes:")
print(res.stdout)

res2 = subprocess.run(["ssh", "root@104.168.91.147", "docker ps 2>/dev/null || true"], capture_output=True, text=True)
print("Docker:")
print(res2.stdout)

res3 = subprocess.run(["ssh", "root@104.168.91.147", "find / -name 'web.api:app' 2>/dev/null || find / -name 'institutional_engine.py' 2>/dev/null"], capture_output=True, text=True)
print("Found Files:")
print(res3.stdout)
