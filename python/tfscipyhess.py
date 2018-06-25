from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.opt import ExternalOptimizerInterface

from scipy.optimize import SR1, LinearConstraint, NonlinearConstraint, Bounds


__all__ = ['ScipyTROptimizerInterface']

def jacobian(ys,
             xs,
             name="hessians",
             colocate_gradients_with_ops=False,
             gate_gradients=False,
             aggregation_method=None):
  """Constructs the jacobian of sum of `ys` with respect to `x` in `xs`.
  `jacobians()` adds ops to the graph to output the Hessian matrix of `ys`
  with respect to `xs`.  It returns a list of `Tensor` of length `len(xs)`
  where each tensor is the jacobian of `sum(ys)`.
  The Hessian is a matrix of second-order partial derivatives of a scalar
  tensor (see https://en.wikipedia.org/wiki/Hessian_matrix for more details).
  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'hessians'.
    colocate_gradients_with_ops: See `gradients()` documentation for details.
    gate_gradients: See `gradients()` documentation for details.
    aggregation_method: See `gradients()` documentation for details.
  Returns:
    A list of jacobian matrices of `sum(ys)` for each `x` in `xs`.
  Raises:
    LookupError: if one of the operations between `xs` and `ys` does not
      have a registered gradient function.
  """
  kwargs = {
      "colocate_gradients_with_ops": colocate_gradients_with_ops,
      "gate_gradients": gate_gradients,
      "aggregation_method": aggregation_method
  }
  # Compute first-order derivatives and iterate for each x in xs.
  #hessians = []
  #_gradients = gradients(ys, xs, **kwargs)
  gradient = ys
  x = xs
  #for gradient, x in zip(_gradients, xs):  
  # change shape to one-dimension without graph branching
  gradient = array_ops.reshape(gradient, [-1])

  # Declare an iterator and tensor array loop variables for the gradients.
  n = array_ops.size(x)
  loop_vars = [
      array_ops.constant(0, dtypes.int32),
      tensor_array_ops.TensorArray(x.dtype, n)
  ]
  # Iterate over all elements of the gradient and compute second order
  # derivatives.
  _, hessian = control_flow_ops.while_loop(
      lambda j, _: j < n,
      lambda j, result: (j + 1,
                          result.write(j, tf.gradients(gradient[j], x)[0])),
      loop_vars
  )

  _shape = array_ops.shape(x)
  _reshaped_hessian = array_ops.reshape(hessian.stack(),
                                        array_ops.concat((_shape, _shape), 0))
  #hessians.append(_reshaped_hessian)
  return _reshaped_hessian

class ScipyTROptimizerInterface(ExternalOptimizerInterface):

  _DEFAULT_METHOD = 'trust-constr'


  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                packed_bounds, step_callback, optimizer_kwargs):

    optimizer_kwargs = dict(optimizer_kwargs.items())
    method = optimizer_kwargs.pop('method', self._DEFAULT_METHOD)
    hess = optimizer_kwargs.pop('hess', SR1())

    constraints = []
    for func, grad_func, tensor in zip(equality_funcs, equality_grad_funcs,self._equalities):
      lb = np.zeros(tensor.shape,dtype=initial_val.dtype)
      ub = lb
      constraints.append(NonlinearConstraint(func, lb, ub, jac = grad_func, hess=SR1()))
    for func, grad_func, tensor in zip(inequality_funcs, inequality_grad_funcs,self._inequalities):
      lb = np.zeros(tensor.shape,dtype=initial_val.dtype)
      ub = np.inf*np.ones(tensor.shape,dtype=initial_val.dtype)
      constraints.append(NonlinearConstraint(func, lb, ub, jac = grad_func, hess=SR1(),keep_feasible=False))

    import scipy.optimize  # pylint: disable=g-import-not-at-top

    if packed_bounds != None:
      lb = np.zeros_like(initial_val)
      ub = np.zeros_like(initial_val)
      for ival,(lbval,ubval) in enumerate(packed_bounds):
        lb[ival] = lbval
        ub[ival] = ubval
      isnull = np.all(np.equal(lb,-np.inf)) and np.all(np.equal(ub,np.inf))
      if not isnull:
        constraints.append(LinearConstraint(np.eye(initial_val.shape[0],dtype=initial_val.dtype),lb,ub,keep_feasible=True))

    minimize_args = [loss_grad_func, initial_val]
    minimize_kwargs = {
        'jac': True,
        'hess' : hess,
        'callback': None,
        'method': method,
        'constraints': constraints,
        'bounds': None,
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
