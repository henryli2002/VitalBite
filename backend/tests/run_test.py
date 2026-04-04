import subprocess
import os

env = os.environ.copy()
# replace CONCURRENT_USERS
with open('tests/load_test_ws.py', 'r') as f:
    code = f.read().replace('CONCURRENT_USERS = 200', 'CONCURRENT_USERS = 5')
with open('tests/load_test_ws_temp.py', 'w') as f:
    f.write(code)

subprocess.run(['python3', 'tests/load_test_ws_temp.py'])
