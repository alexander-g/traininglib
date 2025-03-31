import typing as tp
import time, os, sys
import argparse

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
        y  = self.module(x, *a, **kw)
        y  = self.postprocess(y, x0)
        return y
    
    def process_image(
        self, 
        image:str|np.ndarray, 
        progress_callback:tp.Optional[tp.Callable] = None
    ) -> tp.Any:
        """Full inference pipeline for a single image, from file to result."""
        self.eval()

        x_batches, x0 = self.prepare_image(image)
        with torch.no_grad():
            y_batches: tp.List[torch.Tensor] = []
            for i, batch in enumerate(x_batches):
                y_batches += [self(batch)]
                if callable(progress_callback):
                    progress_callback( (i+1) / len(x_batches) )
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
        assert len(raw) == 1, NotImplementedError('Custom output batch handling required')
        return raw[0].cpu().numpy()[0]
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Input preprocessing function. For both training as well as inference."""
        assert len(x.shape) == 4, 'Input to preprocess() should be batched'
        if x.shape[2] != self.inputsize or x.shape[3] != self.inputsize:
            x = datalib.resize_tensor(x, self.inputsize, "bilinear")
        x = x.to(self.device).to(self.dtype)
        return x

    def postprocess(self, raw: tp.Any, x: torch.Tensor) -> tp.Any:
        """Output postprocessing.
           `x`: original input image before preprocessing"""
        return raw

    @property
    def device(self) -> torch.device:
        """Convenience property to get the device of the task/model"""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    @property
    def dtype(self) -> torch.dtype:
        """Convenience property to get the dtype of the model (float32/float16)"""
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32
    
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
                "torchvision._meta_registrations",
            ]
            pe.intern("torchvision.**", exclude=externs)
            pe.extern(externs)

            pe.save_pickle("model", "model.pkl", self.cpu().eval())
            # pe.save_text('model', 'class_list.txt', '\n'.join(self.class_list))
        return destination
    
    def _start_training(
        self,
        TrainingTask: tp.Type,
        trainsplit:   tp.List[tp.Any],
        valsplit:     tp.List[tp.Any]|None = None,
        *,
        task_kw:      tp.Dict[str, tp.Any] = {},
        fit_kw:       tp.Dict[str, tp.Any] = {},
    ):
        '''Internal method to start a training session.'''
        assert len(trainsplit) > 0
        task = TrainingTask(self, **task_kw)
        return task.fit(trainsplit, valsplit, **fit_kw)
    
    def start_training(
        self,
        trainsplit: tp.List[tp.Any],
        valsplit:   tp.List[tp.Any]|None = None,
        *,
        task_kw:    tp.Dict[str, tp.Any] = {},
        fit_kw:     tp.Dict[str, tp.Any] = {},
    ):
        '''Abstract public interface to start a training session.
           Subclasses should implement it calling super()._start_training() 
           and providing the task and dataset classes'''
        raise NotImplementedError()


def start_training_from_cli_args(
    args:       argparse.Namespace,
    model:      BaseModel, 
    trainsplit: tp.List[tp.Any],
    valsplit:   tp.List[tp.Any]|None  = None,
    task_kw:    tp.Dict[str, tp.Any]  = {},
    fit_kw:     tp.Dict[str, tp.Any]  = {},
) -> bool:
    '''`BaseModel.start_training()` with basic config provided by
       command line arguments from `args.base_training_argparser()`'''
    model, paths = util.prepare_for_training(model, args) # type: ignore
    fit_kw  = {
        'epochs':          args.epochs,
        'lr':              args.lr,
        'batch_size':      args.batchsize,
        'checkpoint_dir':  paths.checkpointdir, 
        'device':          args.device,
    } | fit_kw

    model.start_training(
        trainsplit = trainsplit, 
        valsplit   = valsplit,
        task_kw    = task_kw,
        fit_kw     = fit_kw,
    )
    model.save(paths.modelpath)
    if paths.modelpath_tmp is not None:
        os.remove(paths.modelpath_tmp)
    return True


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

