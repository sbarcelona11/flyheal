"""Isolated CPU worker with terminal job records and parent-lifetime monitoring."""
import json
import os
import sys
import threading
from pathlib import Path
from .training import write_json


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    output = Path(sys.argv[2])
    parent_pid = int(os.environ.get('FLYHEAL_PARENT_PID', os.getppid()))
    stopped = threading.Event()

    def finish(status, error=None):
        job = json.loads((output / 'job.json').read_text())
        job.update(status=status, worker_pid=os.getpid())
        if error:
            job['error'] = error
        write_json(output / 'job.json', job)

    def watch_parent():
        while not stopped.wait(.5):
            if os.getppid() != parent_pid:
                finish('cancelled', 'El servidor que inició el entrenamiento se cerró inesperadamente')
                os._exit(130)

    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        from .experiment import run
        from .plots import plot_results
        from .report import build_report
        run(config, output)
        plot_results(output)
        build_report(output)
        progress = json.loads((output / 'progress.json').read_text())
        progress.update(percent=100, condition='complete')
        write_json(output / 'progress.json', progress)
        finish('complete')
    except BaseException as error:
        finish('failed', str(error))
        raise
    finally:
        stopped.set()


if __name__ == '__main__':
    main()
