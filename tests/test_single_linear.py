import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import jax.numpy as jnp
import torch
import jax


x = np.random.randn(2, 4).astype(np.float64)
w = np.random.randn(4, 8).astype(np.float64)

torch_out = torch.tensor(x) @ torch.tensor(w)
# jax_out = jax.numpy.asarray(x) @ jax.numpy.asarray(w)

# print(np.max(np.abs(torch_out.numpy() - np.array(jax_out))))  # Should be ~<1e-7
jax_out1 = jnp.asarray(x) @ jnp.asarray(w)
print("JAX @      :", np.max(np.abs(torch_out.numpy() - np.array(jax_out1))))

jax_out2 = jnp.matmul(jnp.asarray(x), jnp.asarray(w))
print("jnp.matmul :", np.max(np.abs(torch_out.numpy() - np.array(jax_out2))))

jax_out3 = jnp.dot(jnp.asarray(x), jnp.asarray(w))
print("jnp.dot    :", np.max(np.abs(torch_out.numpy() - np.array(jax_out3))))

from jax import lax

lhs = jnp.asarray(x)
rhs = jnp.asarray(w)
dimension_numbers = (((1,), (0,)), ((), ()))  # Contract on inner dims
jax_out4 = lax.dot_general(lhs, rhs, dimension_numbers)
print("lax.dot_general:", np.max(np.abs(torch_out.numpy() - np.array(jax_out4))))


from jax._src.core import Primitive
print([p for p in Primitive.__subclasses__()])

import jax.lax as lax

print(lax.dot_general_p)
# ➜ Primitive(dot_general)

print(type(lax.dot_general_p))   # <class 'jax.core.Primitive'>
print(lax.dot_general_p.name)   