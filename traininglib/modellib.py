import typing as tp
import time, os, sys
import numpy as np
import torch, torchvision

from . import datalib, util


class BaseModel(torch.nn.Module):
    def __init__(self, module:torch.nn.Module, inputsize:int):
        super().__init__()

        self.module     = module
        self.inputsize  = inputsize
    
    def forward(self, x:torch.Tensor, *a, **kw) -> torch.Tensor:
        x0 = x
        x  = self.preprocess(x)
        y  = self.module(x)
        y  = self.postprocess(y, x0)
        return y
    
    def process_image(self, image: str | np.ndarray) -> tp.Any:
        """Full inference pipeline for a single image, from file to result."""
        self.eval()

        x_batches, x0 = self.prepare_image(image)
        with torch.no_grad():
            y_batches: tp.List[torch.Tensor] = []
            for batch in x_batches:
                y_batches += [self(batch)]
        y = self.finalize_inference(y_batches, x0)
        return y
    
    def prepare_image(self, image: str|np.ndarray) -> tp.Tuple[tp.List[torch.Tensor], torch.Tensor]:
        '''Load or convert image, return list of batches and the unmodified input.
           For inference only.'''
        x = x0 = datalib.ensure_imagetensor(image)
        x = x[np.newaxis]
        return [x], x0
    
    def finalize_inference(self, raw:tp.List[torch.Tensor], x:torch.Tensor) -> np.ndarray:
        '''Convert raw batched outputs to the final result.
           x: original input image before preprocessing'''
        assert len(raw) == 1
        return raw[0].cpu().numpy()[0]
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Input preprocessing function. For both training as well as inference."""
        assert len(x.shape) == 4, 'Input to preprocess() should be batched'
        x = datalib.resize_tensor(x, self.inputsize, "bilinear")
        x = x.to(self.device).to(self.dtype)
        return x

    def postprocess(self, raw: tp.Any, x: torch.Tensor) -> tp.Any:
        """Output postprocessing.
           x: original input image before preprocessing"""
        return raw

    @property
    def device(self) -> torch.device:
        """Convenience property to get the device of the task/model"""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Convenience property to get the dtype of the model (float32/float16)"""
        return next(self.parameters()).dtype
    
    def save(self, destination: str) -> str:
        """Save a model as a self-contained torch.package including source code."""
        if isinstance(destination, str):
            destination = time.strftime(destination)
            if not destination.endswith(".pt.zip"):
                destination += ".pt.zip"
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)

        try:
            import torch_package_importer as imp

            # re-export
            importer: tp.Any = (imp, torch.package.sys_importer)
        except ImportError as e:
            # first export
            importer = (torch.package.sys_importer,)

        with torch.package.PackageExporter(destination, importer) as pe:
            # save all python files in src folder
            interns = util.collect_loaded_non_venv_modules()
            pe.intern(interns)
            pe.extern("**", exclude=["torchvision.**"])
            externs = [
                "torchvision.ops.**",
                "torchvision.datasets.**",
                "torchvision.io.**",
            ]
            pe.intern("torchvision.**", exclude=externs)
            pe.extern(externs)

            pe.save_pickle("model", "model.pkl", self.cpu().eval())
            # pe.save_text('model', 'class_list.txt', '\n'.join(self.class_list))
        return destination
    
    def start_training(
        self,
        trainsplit:   tp.List[tp.Any],
        DatasetClass: tp.Type,
        TrainingTask: tp.Type,
        ds_kw:        tp.Dict[str, tp.Any]        = {},
        ld_kw:        tp.Dict[str, tp.Any]        = {},
        task_kw:      tp.Dict[str, tp.Any]        = {},
        fit_kw:       tp.Dict[str, tp.Any]        = {},
    ):
        assert len(trainsplit) > 0
        ds    = DatasetClass(trainsplit, **ds_kw)
        ld_kw = {'batch_size':4} | ld_kw
        ld    = datalib.create_dataloader(ds, shuffle=True, **ld_kw)
        print(f"Training on {len(ds)} images / {len(ld)} batches.")
        task  = TrainingTask(self, **task_kw)
        return task.fit(ld, **fit_kw)


def load_model(filename:str) -> BaseModel:
    '''Load a self-contained torch.package from file as saved with .save() above'''
    assert os.path.exists(filename), filename
    return torch.package.PackageImporter(filename).load_pickle('model', 'model.pkl')

def load_weights(filepath:str, model:torch.nn.Module) -> None:
    if filepath.endswith('.pt.zip'):
        sd = load_model(filepath).state_dict()
    elif filepath.endswith('.pth'):
        sd = torch.load(filepath)
    else:
        raise NotImplementedError(f"Don't know how to load weights from {filepath}")
    
    return model.load_state_dict(sd)

