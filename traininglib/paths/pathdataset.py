import os
import typing as tp

import numpy as np
import PIL.Image
import torch

from .. import datalib
from ..segmentation import PatchedCachingDataset, load_if_cached
from . import svg

FilePair:tp.TypeAlias = datalib.FilePair


class PatchedPathsDataset(PatchedCachingDataset):
    def __init__(self, *a, first_component_only:bool, **kw):
        '''`first_component_only = True` will only use the first component of a
           path instance (e.g. only the main root in arabidopsis roots)'''
        self.first_component_only = first_component_only
        super().__init__(*a, **kw)

    def _cache(self, filepairs, cachedir):
        #inputfiles = [imf for imf,_ in filepairs]
        svgfiles   = [anf for _,anf in filepairs]

        cached_filepairs, grids = \
            super()._cache(filepairs, prefixes=['in'], cachedir=cachedir)
        if grids is None:
            #cached
            assert len(cached_filepairs[0]) == 2
            return cached_filepairs, None

        cached_inputfiles = [pair[0] for pair in cached_filepairs]
        # concrete cachedir
        cachedir = os.path.dirname(cached_inputfiles[0].replace('/in/','/an/'))
        os.makedirs(cachedir, exist_ok=True)

        svg_items = []
        for i,grid in enumerate(grids):
            parsed = svg.parse_svg(svgfiles[i])
            assert parsed.size == tuple(grid[-1][-2:][::-1]), \
                'Image and annotation have different sizes'
            if self.first_component_only:
                # only using the first component (the main root)
                components = [np.array(pathlist[0]) for pathlist in parsed.paths]
            else:
                # using all components
                components = [
                    np.array(path) 
                        for pathlist in parsed.paths 
                            for path in pathlist
                ]
            
            if getattr(self, 'scale', 1.0) != 1.0:
                components = [c * self.scale for c in components]

            for j,coords in enumerate(grid):
                y0,x0 = topleft     = grid.reshape(-1,4)[j,:2][::-1]
                y1,x1 = bottomright = grid.reshape(-1,4)[j,-2:][::-1]

                filtered_components = [
                    c for c in components 
                        if np.any( np.all(c >= topleft, axis=-1) )
                        and np.any( np.all(c <= bottomright, axis=-1 ) )
                ]
                shifted_components:tp.List[svg.Path] = [
                    [component - topleft] for component in filtered_components
                ]

                size    = bottomright - topleft
                svg_str = svg.export_paths_as_svg(shifted_components, size)
                svg_dst = os.path.join(
                    cachedir, f'{os.path.basename(svgfiles[i])}.{j:04d}.svg'
                )
                open(svg_dst, 'w').write(svg_str)
                svg_items.append(svg_dst)
        

        assert len(svg_items) == len(cached_inputfiles)
        cached_inputfiles = [os.path.abspath(p) for p in cached_inputfiles]
        svg_items = [os.path.abspath(p) for p in svg_items]
        new_filepairs = list(zip(cached_inputfiles, svg_items))

        cachefile = os.path.join(cachedir, '..', 'cachefile.csv')
        datalib.save_file_tuples(cachefile, new_filepairs)
        return new_filepairs, grids

    def __getitem__(self, i:int) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        inputfile, svgfile = self.filepairs[i]
        inputdata = datalib.load_image(inputfile, to_tensor=True)
        parsed    = svg.parse_svg(svgfile)
        if self.first_component_only:
            components = [np.array(pathlist[0]) for pathlist in parsed.paths]
        else:
            components = [
                np.array(path) 
                    for pathlist in parsed.paths 
                        for path in pathlist
            ]
        return inputdata, components # type: ignore

    def collate_fn(self, items:tp.List):
        return items


class FirstComponentPatchedPathsDataset(PatchedPathsDataset):
    def __init__(self, *a, **kw):
        super().__init__(*a, first_component_only=True, **kw)

class AllComponentsPatchedPathsDataset(PatchedPathsDataset):
    def __init__(self, *a, **kw):
        super().__init__(*a, first_component_only=False, **kw)
