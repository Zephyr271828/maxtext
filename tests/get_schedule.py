import jax.numpy as jnp
import optax
# import matplotlib.pyplot as plt
import json

# Define schedule parameters
total_steps = 12500
warmup_steps = 625
lr = 3e-4
final_lr_ratio = 0.1  # cosine final LR = lr * final_lr_ratio

# Define warmup and cosine schedules
warmup = optax.polynomial_schedule(
    init_value=0.0,
    end_value=lr,
    power=2.0,
    transition_steps=warmup_steps
)

cosine = optax.cosine_decay_schedule(
    init_value=lr,
    decay_steps=total_steps,
    alpha=final_lr_ratio
)

# Define FMS-style schedule: min(warmup, cosine)
def fms_style_lr(step):
    return jnp.minimum(warmup(step), cosine(step))

res = [float(fms_style_lr(step)) for step in range(0, 12501)]

# print(res[:100])

with open('lr_schedule.json', 'w') as f:
    json.dump(res, f)