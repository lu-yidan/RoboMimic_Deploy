"""Wait for the oracle fine-tune to finish, then run the 24-case evaluation automatically."""
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(r'D:\G1Climbing\external\RoboMimic_Deploy\outputs\calibrated_geometry_ppo131072_20261029_oracle_v1')
EVAL = Path(r'D:\G1Climbing\.dsh-watch\oracle_all24.py')
LOG = Path(r'D:\G1Climbing\.dsh-watch\oracle_wait_eval.log')
PY = Path(r'D:\G1Climbing\external\RoboMimic_Deploy\.venv\Scripts\python.exe')

deadline = time.time() + 3 * 3600


def log(message):
    stamp = time.strftime('%H:%M:%S')
    with LOG.open('a', encoding='utf-8') as stream:
        stream.write('[%s] %s\n' % (stamp, message))


log('watching %s' % RUN)
while time.time() < deadline:
    summary = RUN / 'summary.json'
    if summary.exists() and summary.stat().st_size > 1000:
        log('training finished; launching evaluation')
        with LOG.open('a', encoding='utf-8') as stream:
            subprocess.run([str(PY), str(EVAL)], stdout=stream, stderr=subprocess.STDOUT,
                           cwd=str(RUN.parents[1]))
        log('evaluation done')
        sys.exit(0)
    time.sleep(60)

log('timeout: training did not finish within 3 h')
sys.exit(1)
