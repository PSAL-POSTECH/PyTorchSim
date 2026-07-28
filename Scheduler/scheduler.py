"""Poisson load helpers for synthetic request arrival times."""

import numpy as np


def poisson_request_generator(lambda_requests, max_msec_time=None):
    """Yield synthetic arrival times in milliseconds (first sample is 0)."""
    current_time = 0.0  # msec

    yield 0
    while max_msec_time is None or current_time < max_msec_time:
        inter_arrival_time = np.random.exponential(scale=1000 / lambda_requests)
        current_time += inter_arrival_time

        if max_msec_time is not None and current_time > max_msec_time:
            break

        yield current_time
