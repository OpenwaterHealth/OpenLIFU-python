# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: env
#     language: python
#     name: python3
# ---

import numpy as np

import openlifu

pulse = openlifu.Pulse(frequency=500e3, amplitude=1, duration=2e-5)
pt = openlifu.Point(position=(0,0,30), units="mm")
sequence = openlifu.Sequence(
    pulse_interval=0.1,
    pulse_count=10,
    pulse_train_interval=1,
    pulse_train_count=1
)
solution = openlifu.Solution(
    id="solution",
    name="Solution",
    protocol_id="example_protocol",
    transducer_id="example_transducer",
    delays = np.zeros((1,64)),
    apodizations = np.ones((1,64)),
    pulse = pulse,
    sequence = sequence,
    target=pt,
    foci=[pt],
    approved=True
    )


solution
