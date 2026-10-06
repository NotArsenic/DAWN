import re
with open("tests/test_camera.py", "r") as f:
    text = f.read()

text = re.sub(r'        filtered =', '    filtered =', text)

with open("tests/test_camera.py", "w") as f:
    f.write(text)

