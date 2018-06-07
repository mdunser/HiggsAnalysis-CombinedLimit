from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.opt import ExternalOptimizerInterface

from scipy.optimize import SR1
from scipy.optimize import NonlinearConstraint


__all__ = ['ScipyTROptimizerInterface']

class ScipyTROptimizerInterface(ExternalOptimizerInterface):

  _DEFAULT_METHOD = 'trust-constr'


  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                packed_bounds, step_callback, optimizer_kwargs):

    def loss_grad_func_wrapper(x):
      # SciPy's L-BFGS-B Fortran implementation requires gradients as doubles.
      loss, gradient = loss_grad_func(x)
      return loss, gradient.astype('float64')

    optimizer_kwargs = dict(optimizer_kwargs.items())
    method = optimizer_kwargs.pop('method', self._DEFAULT_METHOD)
    hess = optimizer_kwargs.pop('hess', SR1())

    constraints = []
    for func, grad_func in zip(equality_funcs, equality_grad_funcs):
      constraints.append(NonlinearConstraint(func, 0.,0.,jac = grad_func, hess=SR1()))
    for func, grad_func in zip(inequality_funcs, inequality_grad_funcs):
      constraints.append(NonlinearConstraint(func, 0.,np.inf,jac = grad_func, hess=SR1()))

    import scipy.optimize  # pylint: disable=g-import-not-at-top

    minimize_args = [loss_grad_func_wrapper, initial_val]
    minimize_kwargs = {
        'jac': True,
        'hess' : hess,
        'callback': None,
        'method': method,
        'constraints': constraints,
        'bounds': packed_bounds,
    }

    for kwarg in minimize_kwargs:
      if kwarg in optimizer_kwargs:
        if kwarg == 'bounds':
          # Special handling for 'bounds' kwarg since ability to specify bounds
          # was added after this module was already publicly released.
          raise ValueError(
              'Bounds must be set using the var_to_bounds argument')
        raise ValueError(
            'Optimizer keyword arg \'{}\' is set '
            'automatically and cannot be injected manually'.format(kwarg))

    minimize_kwargs.update(optimizer_kwargs)

    
    result = scipy.optimize.minimize(*minimize_args, **minimize_kwargs)

    message_lines = [
        'Optimization terminated with:',
        '  Message: %s',
        '  Objective function value: %f',
    ]
    message_args = [result.message, result.fun]
    if hasattr(result, 'nit'):
      # Some optimization methods might not provide information such as nit and
      # nfev in the return. Logs only available information.
      message_lines.append('  Number of iterations: %d')
      message_args.append(result.nit)
    if hasattr(result, 'nfev'):
      message_lines.append('  Number of functions evaluations: %d')
      message_args.append(result.nfev)
    logging.info('\n'.join(message_lines), *message_args)

    return result['x']
