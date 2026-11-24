"""Read bridge DDS only; export time-stamped waist history without motor publishers."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from onboard.localization.dlio.core import WaistHistory, atomic_json
from policy.acp.state_ipc import boot_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/dev/shm/robomimic-%d/acp-waist.json' % os.getuid())
    parser.add_argument('--domain', type=int, default=0)
    args = parser.parse_args()
    path = Path(args.output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from bridge.python.bridge_state_dds import BridgeStateSubscriber
        subscriber = BridgeStateSubscriber(domain_id=args.domain)
        history, identity = WaistHistory(), str(uuid.uuid4())
        subscriber.start()
        written = 0.
        print('Waist export only: no command publisher. Requires a running C++ bridge.')
        try:
            while True:
                state = subscriber.latest()
                now = time.monotonic()
                stamp = getattr(state, '_local_received_monotonic', 0.)
                if state.tick and 0 <= now-stamp < .15:
                    # Repeated SDK ticks never refresh the capture time.
                    history.add(int(state.tick), stamp, state.q)
                    if now-written >= .01 and history.rows:
                        atomic_json(path, dict(schema=1, boot_id=boot_id(), source_id=identity, rows=history.rows))
                        written = now
                time.sleep(.002)
        except KeyboardInterrupt:
            pass
        finally:
            subscriber.stop()
            path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
