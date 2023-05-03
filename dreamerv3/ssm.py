from . import ninjax as nj
from .ssm.s4 import S4Block
from .nets import SSM


class S4SSM(SSM):
  def __init__(self, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
    unimix=0.01, action_clip=1.0, new_form=False, **kw):
    self.cell = nj.FlaxModule(
      S4Block, 
      layer={'N': 128, 'l_max': 255},
      d_model=deter, dropout=0.0, decode=True
    )
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._initial = initial
    self._unimix = unimix
    self._new_form = new_form
    self._action_clip = action_clip
    self._kw = kw

  def get_dist(self, state, argmax=False):
    return NotImplemented

  def _cell(self, x, deter):
    pass

  def get_stoch(self, deter):
    return NotImplemented