import sys

print("Python executable:", sys.executable)
print("sys.path[0]:", sys.path[0])
print("site-packages in:", [p for p in sys.path if "site-packages" in p][:3])