'''
quickzonoreach profiling example

this produces the tables in the readme

Stanley Bak, Feb 2020
'''

import time
import math

import numpy as np
from quickzonoreach.zono import get_zonotope_reachset

def run_single_profile(dims, num_steps, quick=False, save_all=True):
    '''run the computation and return zonos for a single parameter setup

    if store_all is False, only last result is stored

    returns zono list if save_all is true, or last zonotope otherwise
    '''

    np.random.seed(0)

    # dynamics is noisy + harmonic oscillator for every two dimensions
    # x' = y + u1, y' = -x + + u1 + u2
    noise = 0.05
    a_mat = np.random.random((dims, dims)) * (2*noise) - noise
    b_mat = np.zeros((dims, 2), dtype=float)

    a_mat_one = [[0, 1], [-1, 0]]
    b_mat_one = [[1, 0], [1, 1]]

    init_box_one = [[-5, -4], [0, 1]]
    input_box = [[-0.5, 0.5], [-1, 0]]

    init_box = []

    tmax = math.pi
    dt = tmax / num_steps

    assert dims % 2 == 0, "expected even number of dimensions"

    for d in range(dims // 2):
        a_mat[2*d:2*d+2, 2*d:2*d+2] = a_mat_one
        b_mat[2*d:2*d+2, 0:2] = b_mat_one

        init_box += init_box_one

    a_mat_list = []
    b_mat_list = []
    input_box_list = []
    dt_list = []
    save_list = []

    for _ in range(num_steps):
        a_mat_list.append(a_mat)
        b_mat_list.append(b_mat)
        input_box_list.append(input_box)
        dt_list.append(dt)
        save_list.append(save_all)

    # always save last one
    save_list.append(True)

    zonotopes = get_zonotope_reachset(init_box, a_mat_list, b_mat_list, input_box_list, dt_list, \
                                      save_list=save_list, quick=quick)

    if save_all:
        rv = zonotopes
    else:
        assert len(zonotopes) == 1, f"zonotopes len was {len(zonotopes)}"
        rv = zonotopes[0]

    return rv

def main():
    '''generate timing statistics and print to stdout

    This evaluates scalability as the size of a_mat increase and the number of steps increases.
    '''

    max_time = 1.0
    start_steps = 8
    print(f"Profiling with timeout {max_time} (expected ~80 seconds)...")
    total_start = time.perf_counter()

    # params are 2-tuple: (quick, save_all)
    params = [(False, True), (True, True), (True, False)]

    for quick, save_all in params:
        print("")
        dims = 2

        if quick and not save_all:
            start_steps = 32
        else:
            start_steps = 8

        data = []
        stop = False

        while not stop:
            row = [f'**{dims} dims**']
            num_steps = start_steps

            while True:
                start = time.perf_counter()
                run_single_profile(dims, num_steps, quick, save_all=save_all)
                diff = time.perf_counter() - start

                print(f"dims: {dims}, steps: {num_steps}, time: {round(1000 * diff, 1)}ms")

                if diff > 0.1:
                    res = round(diff, 1)
                elif diff > 0.01:
                    res = round(diff, 2)
                elif diff > 0.001:
                    res = round(diff, 3)
                else:
                    res = round(diff, 4)
                    
                row.append(str(res))

                if diff > max_time:
                    break

                num_steps *= 2

            if not data:
                # append steps
                step_str_list = [f"**{start_steps * 2**(count)} steps**" for count in range(len(row)-1)]

                if not quick:
                    label = "Exact Save All"
                elif save_all:
                    label = "Quick Save All"
                else:
                    label = "Quick Save Last"
                    
                data.append([label] + step_str_list)

            data.append(row)

            if len(row) == 2: # single entry (After label)
                stop = True

            # fill in row to match original length
            while len(row) < len(data[0]):
                row.append("-")

            dims *= 2

        print(' | '.join(data[0]))
        dashes = ['---' for _ in data[0]]
        print(' | '.join(dashes))

        for row in data[1:]:
            print(' | '.join(row))

    diff = time.perf_counter() - total_start
    print(f"Completed with timeout {max_time} in {round(diff, 1)} secs")

if __name__ == "__main__":
    main()
