import io
import torch
import onnxruntime as ort


def _export_to_onnx(func, args=None, dyn_ax=None, **kw):
    buffer   = io.BytesIO()
    torch.onnx.export(
        torch.jit.script(func), 
        args or (torch.zeros([1,1,64,64]),), 
        buffer, 
        input_names=['x'], 
        dynamic_axes=dyn_ax or {'x':[2,3]},
    )
    onnx_bytes = buffer.getvalue()

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(onnx_bytes, sess_options)
    return session
