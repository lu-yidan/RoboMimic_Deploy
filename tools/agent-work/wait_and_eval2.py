"""Wait for a training run to finish, then run its 24-case evaluation automatically.

usage: wait_and_eval2.py <run directory> <eval script>
"""
import subprocess
import sys
import time
from pathlib import Path

PY = Path(r'D:\G1Climbing\external\RoboMimic_Deploy\.venv\Scripts\python.exe')
LOG = Path(r'D:\G1Climbing\.dsh-watch\wait_eval2.log')

run = Path(sys.argv[1])
eval_script = Path(sys.argv[2])
deadline = time.time() + 6 * 3600


def log(message):
    with LOG.open('a', encoding='utf-8') as stream:
        stream.write('[%s] %s\n' % (time.strftime('%H:%M:%S'), message))


log('watching %s -> %s' % (run, eval_script.name))
while time.time() < deadline:
    summary = run / 'summary.json'
    if summary.exists() and summary.stat().st_size > 1000:
        log('training finished; launching evaluation')
        with LOG.open('a', encoding='utf-8') as stream:
            subprocess.run([str(PY), str(eval_script)] + sys.argv[3:], stdout=stream, stderr=subprocess.STDOUT,
                           cwd=str(Path(r'D:\G1Climbing\external\RoboMimic_Deploy')))
        log('evaluation done')
        sys.exit(0)
    time.sleep(60)

log('timeout')
sys.exit(1)
