import warnings
warnings.simplefilter('ignore', UserWarning) #pytorch is annoying

import os, sys
from contextlib import contextmanager

@contextmanager
def silence_std():
    try:
        out_fd = sys.stdout.fileno()
        err_fd = sys.stderr.fileno()
    except Exception:
        yield; return
    sys.stdout.flush(); sys.stderr.flush()
    saved_out = os.dup(out_fd); saved_err = os.dup(err_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, out_fd); os.dup2(devnull, err_fd)
    os.close(devnull)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = os.fdopen(os.dup(out_fd), 'w')
    sys.stderr = os.fdopen(os.dup(err_fd), 'w')
    try:
        yield
    finally:
        try: sys.stdout.flush(); sys.stderr.flush()
        except: pass
        try: sys.stdout.close(); sys.stderr.close()
        except: pass
        sys.stdout, sys.stderr = old_out, old_err
        os.dup2(saved_out, out_fd); os.dup2(saved_err, err_fd)
        os.close(saved_out); os.close(saved_err)

with silence_std():
    try:
        # onnxruntime is also annoying
        import onnxruntime
    except ImportError:
        pass


