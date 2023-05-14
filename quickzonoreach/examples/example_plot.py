'''
quickzonoreach plotting example

Stanley Bak, Feb 2020
'''

import math
import sys

import matplotlib.pyplot as plt

sys.path.append("/home/kai/Desktop/quickzonoreach")
from quickzonoreach.zono import get_zonotope_reachset

def main():
    'example to make quickzonoreach.png'

    filename = 'quickzonoreach.png'
    print(f"making {filename}...")
    
    # x' = y + u1, y' = -x + + u1 + u2
    # u1 in [-0.5, 0.5], u2 in [-1, 0]
    
    a_mat = [[0, 1], [-1, 0]]
    b_mat = [[1, 0], [1, 1]]

    init_box = [[-5, -4], [0, 1]]
    input_box = [[-0.5, 0.5], [-1, 0]]

    num_steps = 4
    dt = math.pi / num_steps

    a_mat_list = []
    b_mat_list = []
    input_box_list = []
    dt_list = []

    for _ in range(num_steps):
        a_mat_list.append(a_mat)
        b_mat_list.append(b_mat)
        input_box_list.append(input_box)
        dt_list.append(dt)

    zonotopes = get_zonotope_reachset(init_box, a_mat_list, b_mat_list, input_box_list, dt_list)

    # plot first set in red
    plt.figure(figsize=(6, 6))
        
    zonotopes[0].plot(col='r-o', label='Init')

    for i, z in enumerate(zonotopes[1:]):
        label = 'Reach Set' if i == 0 else None
        z.plot(label=label)

    plt.title('Quickzonoreach Output (example_plot.py)')
    plt.legend()
    plt.grid()
    plt.savefig(filename)

if __name__ == "__main__":
    main()
