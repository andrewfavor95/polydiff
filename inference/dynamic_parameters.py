from collections import namedtuple
from abc import ABC, abstractmethod
from scipy.interpolate import CubicSpline

Step = namedtuple('Step', 'design diffuse rfold')

class DynamicParam(ABC):
    """parameter that can change values through a process"""
    def __init__(self, manager=None):
        self.manager = manager

    def attach_manager(self, manager):
        self.manager = manager

    def __str__(self):
        val = 'No Current Value'
        try:
            val = self.value()
        except TypeError:
            pass
        return f'{self.__class__.__name__} {val}'

    @property
    def progress(self):
        return self.manager.progress()

    @abstractmethod
    def value(self):
        pass

    def __bool__(self):
        return bool(self.value())

    def __float__(self):
        return float(self.value())

    def __int__(self):
        return int(self.value())

class DynamicParameters:
    def __init__(self):
        self.step = None
        self.totstep = None
        self.rfold_tag = None
        self.params = dict()

    def add_param(self, name, param):
        assert not name in self.params
        if isinstance(param, DynamicParam):
            param.attach_manager(self)
        else:
            assert isinstance(param, (bool,int,float))
        self.params[name] = param

    def __getattr__(self, k):
        if k in self.params:
            return self.params[k]
        raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, k))

    def set_progress(self, step, totstep=None):
        self.step = Step(*step)
        if self.totstep is not None:
            assert self.totstep == totstep  # just checking...
        self.totstep = Step(*totstep)
        assert self.step.rfold == -1
        self.rfold_tag = None

    def new_rfold_iter(self, tag):
        assert self.rfold_tag != tag, f'iter recorded twice! {self.step} {tag}'
        self.rfold_tag = tag
        self.step = Step(self.step.design, self.step.diffuse, self.step.rfold + 1)

    @property
    def progress(self):
        return [s / max(1, ts - 1) for s, ts in zip(self.step, self.totstep)]

    def add_hydraconf(self, conf):
        print('TODO get dynp from hydra')

    def __str__(self):
        s = 'DynamicParameters(\n'
        for k, v in self.params.items():
            s += f'   {k}: {v}'
        s += ')'
        if len(s) < 80: s = s.replace('\n', '')
        return s


def _as_set(thing):
    if thing is None: return thing
    try:
        return set(thing)
    except TypeError:
        return [thing]

class TrueOnIters(DynamicParam):
    def __init__(self, diffuse, rfold, design=None):
        super().__init__()
        self.diffuse_steps = _as_set(diffuse)
        self.rfold_steps = _as_set(rfold)
        self.design_steps = _as_set(design)

    def value(self):
        design_step, diffuse_step, rfold_step = self.manager.step
        if self.design_steps is not None and design_step not in self.design_steps: return False
        if self.diffuse_steps is not None and diffuse_step not in self.diffuse_steps: return False
        if self.rfold_steps is not None and rfold_step not in self.rfold_steps: return False
        return True

class FalseOnIters(TrueOnIters):
    def value(self):
        return not super().value()

# class SplineParam(DynamicParam):
#     def __init__(self, xsamp, ysamp):
#         super().__init__()
#         self.interp = CubicSpline(xsamp, ysamp)
# 
#     def __float__(self):
#         return self.interpolator()
