"""Actual scalar optical operators and task dependencies, NumPy/torch CUDA."""
from __future__ import annotations
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


def configuration():
    return json.loads(Path(__file__).with_name('config.json').read_text())


def specification(case):
    tasks = [dict(id=0, op='aperture', inputs=[]), dict(id=1, op='spectrum', inputs=[0])]
    for p in range(case['planes']):
        a = len(tasks)
        tasks.extend([dict(id=a, op='propagate', inputs=[1], plane=p),
                      dict(id=a + 1, op='intensity', inputs=[a]),
                      dict(id=a + 2, op='metrics', inputs=[a + 1])])
    tasks.append(dict(id=len(tasks), op='collect', inputs=[4 + 3*p for p in range(case['planes'])]))
    return tasks


class Operators:
    def __init__(self, n, config, device='cpu'):
        self.n, self.config, self.device = n, config, device
        self.torch = None
        if device == 'cuda':
            import torch
            self.torch = torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required, never silently fall back to CPU')

    def sync(self):
        if self.torch is not None:
            self.torch.cuda.synchronize()

    def array(self, value):
        if self.torch is None:
            return np.asarray(value)
        return self.torch.as_tensor(value, device='cuda')

    def numpy(self, value):
        return np.asarray(value) if self.torch is None else value.detach().cpu().numpy()

    def run(self, task, inputs):
        n, cfg, torch = self.n, self.config, self.torch
        op = task['op']
        if op == 'aperture':
            # Same analytic double-precision input on both resources. Entry is
            # pinned to CPU in all scheduling experiments.
            xy = (np.arange(n, dtype=np.float64) - (n - 1)/2) * cfg['pixel_pitch_m']
            x, y = np.meshgrid(xy, xy, indexing='xy')
            radius = n * cfg['pixel_pitch_m'] * 0.22
            r2 = (x*x + y*y)/(radius*radius)
            field = np.exp(-r2) * (r2 <= 1) * np.exp(1j * (0.6*r2 + 0.15*(x/radius)**3))
            return self.array(field.astype(np.complex128))
        if op == 'spectrum':
            padded = cfg['padding_factor'] * n
            offset = (padded - n)//2
            if torch is None:
                field = np.zeros((padded,padded), dtype=np.complex128)
                field[offset:offset+n,offset:offset+n] = inputs[0]
                return np.fft.fft2(field)
            field = torch.zeros((padded,padded), dtype=torch.complex128, device='cuda')
            field[offset:offset+n,offset:offset+n] = inputs[0]
            return torch.fft.fft2(field)
        if op == 'propagate':
            padded = cfg['padding_factor'] * n
            z = cfg['distance_step_m'] * (task['plane'] + 1)
            k = 2*math.pi/cfg['wavelength_m']
            if torch is None:
                f = 2*math.pi*np.fft.fftfreq(padded, cfg['pixel_pitch_m'])
                kx, ky = np.meshgrid(f, f, indexing='xy')
                kz = np.sqrt((k*k - kx*kx - ky*ky).astype(np.complex128))
                h = np.exp(1j*z*kz)
                field = np.fft.ifft2(inputs[0]*h)
            else:
                f = 2*math.pi*torch.fft.fftfreq(padded, cfg['pixel_pitch_m'], dtype=torch.float64, device='cuda')
                ky, kx = torch.meshgrid(f, f, indexing='ij')
                kz = torch.sqrt((k*k-kx*kx-ky*ky).to(torch.complex128))
                h = torch.exp(1j*z*kz)
                field = torch.fft.ifft2(inputs[0]*h)
            offset = (padded-n)//2
            return field[offset:offset+n,offset:offset+n].clone() if torch is not None else field[offset:offset+n,offset:offset+n].copy()
        if op == 'intensity':
            return inputs[0].real**2 + inputs[0].imag**2
        if op == 'metrics':
            # Pixel-coordinate moments are normalized by integrated intensity;
            # the constant dx² is used for the integrated-intensity measure.
            xy = (np.arange(n,dtype=np.float64)-(n-1)/2)*cfg['pixel_pitch_m']
            x,y = np.meshgrid(xy,xy,indexing='xy')
            r2 = self.array(x*x+y*y)
            mask = self.array(x*x+y*y <= (0.1*n*cfg['pixel_pitch_m'])**2)
            intensity = inputs[0]
            total = intensity.sum()
            if torch is None:
                return np.asarray([total*cfg['pixel_pitch_m']**2, intensity.max(),
                                   intensity[mask].sum()/total, (r2*intensity).sum()/total],dtype=np.float64)
            return torch.stack([total*cfg['pixel_pitch_m']**2,intensity.max(),
                                intensity[mask].sum()/total,(r2*intensity).sum()/total])
        if op == 'collect':
            if torch is None:
                return np.concatenate([np.concatenate(inputs),np.asarray([np.argmin([a[3] for a in inputs])],dtype=np.float64)])
            metrics = torch.stack(inputs)
            return torch.cat([metrics.flatten(),torch.argmin(metrics[:,3]).reshape(1).to(torch.float64)])
        raise ValueError(op)


def summary(value):
    value = np.ascontiguousarray(value)
    flat = value.reshape(-1)
    take = np.unique(np.linspace(0,len(flat)-1,min(64,len(flat))).astype(int))
    return dict(shape=list(value.shape),dtype=str(value.dtype),nbytes=value.nbytes,
                sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                l2=float(np.linalg.norm(flat)),max_abs=float(np.abs(flat).max()),
                sample_indices=take.tolist(),sample_real=np.real(flat[take]).tolist(),
                sample_imag=np.imag(flat[take]).tolist())


def check_numeric(actual,expected,cfg):
    difference = np.asarray(actual)-np.asarray(expected)
    abs_error = float(np.max(np.abs(difference)))
    rel_error = float(np.linalg.norm(difference.reshape(-1))/max(np.linalg.norm(np.asarray(expected).reshape(-1)),1e-300))
    okay = abs_error <= cfg['absolute_tolerance'] and rel_error <= cfg['relative_l2_tolerance']
    if not okay:
        raise RuntimeError(f'numerical equivalence failed: abs={abs_error}, rel_l2={rel_error}')
    return dict(max_absolute_error=abs_error,relative_l2_error=rel_error,passed=True)


def execute(case,cfg,device='cpu'):
    engine = Operators(case['n'],cfg,device)
    results = {}
    for task in specification(case):
        results[task['id']] = engine.run(task,[results[i] for i in task['inputs']])
    engine.sync()
    return engine,results


def profile(case,cfg,device):
    engine = Operators(case['n'],cfg,device)
    reference = None
    if device == 'cuda':
        _,reference = execute(case,cfg,'cpu')
    outputs = {}
    rows = []
    for task in specification(case):
        inputs = [outputs[i] for i in task['inputs']]
        for _ in range(cfg['profile_warmups']):
            engine.run(task,inputs)
        engine.sync()
        samples = []
        for repetition in range(cfg['profile_repetitions']):
            engine.sync()
            start = time.perf_counter()
            value = engine.run(task,inputs)
            engine.sync()
            samples.append(time.perf_counter()-start)
        outputs[task['id']] = value
        array = engine.numpy(value)
        validation = check_numeric(array,reference[task['id']],cfg) if reference is not None else dict(passed=True,reference='local analytic execution')
        rows.append(dict(**task,seconds=samples,median_seconds=float(np.median(samples)),
                         output=summary(array),numerical_validation=validation))
    final = engine.numpy(outputs[len(rows)-1]).tolist()
    return dict(case=case,device=device,tasks=rows,final_metrics=final)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['profile','pilot'])
    parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    cfg=configuration()
    out=Path(args.out)
    if out.exists():
        raise RuntimeError('refusing to overwrite existing profile')
    env=dict(host=platform.node(),os=platform.platform(),python=sys.version,numpy=np.__version__,device=args.device,
             config_sha256=hashlib.sha256(Path(__file__).with_name('config.json').read_bytes()).hexdigest(),
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    if args.device=='cuda':
        import torch
        torch.set_num_threads(1)
        env.update(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda)
    start=time.perf_counter()
    cases=cfg['workloads'][:1] if args.stage=='pilot' else cfg['workloads']
    results=[]
    for case in cases:
        results.append(profile(case,cfg,args.device))
        print(case['id'],'complete',flush=True)
    data=dict(environment=env,profiles=results,wall_seconds=time.perf_counter()-start)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('PROFILE_OK',data['wall_seconds'],flush=True)


if __name__=='__main__':
    main()
