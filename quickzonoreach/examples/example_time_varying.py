'''
quickzonoreach time varying with simulation example

Stanley Bak, Feb 2020
'''

import matplotlib.pyplot as plt

from quickzonoreach.zono import get_zonotope_reachset
from quickzonoreach.sim import get_center_sim

from numpy import array

def main():
    'example to make quickzonoreach.png'

    filename = 'time_varying.png'
    print(f"making {filename}...")

    a_mat_list = [array([[0.        , 0.        , 0.24740396, 0.06185099, 0.        ,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.96891242, 0.        , 0.24222811,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.75696283, 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 1.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        ]]), array([[0.        , 0.        , 0.34878718, 0.10286165, 0.        ,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 1.14819315, 0.        , 0.33861634,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.90410493, 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 1.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        ]]), array([[0.        , 0.        , 0.4724748 , 0.16264505, 0.        ,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 1.31786477, 0.        , 0.45366266,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            1.03993226, 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 1.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        ]])]

    b_mat_list = [array([[ 0.96891242,  0.24740396],
           [-0.24740396,  0.96891242],
           [-0.77313737,  3.02785132],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ]]), array([[ 0.95682762,  0.34878718],
           [-0.29065598,  1.14819315],
           [-0.76935511,  3.63457749],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ]]), array([[ 0.94133198,  0.4724748 ],
           [-0.337482  ,  1.31786477],
           [-0.75803898,  4.24433313],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ]])]

    input_box_list = [array([[ 0.9       ,  1.1       ],
           [-0.28488889, -0.21511111]]), array([[ 1.1       ,  1.3       ],
           [-0.28363993, -0.21386215]]), array([[ 1.3       ,  1.5       ],
           [-0.27990553, -0.21012776]])]

    dt_list = [0.05, 0.05, 0.05]

    init_box = [(-0.1, 0.1), (-0.1, 0.1), (0, 0), (1, 1), (1, 1), (1, 1), (0.0, 0.0), (1.0, 1.0)]

    zonotopes = get_zonotope_reachset(init_box, a_mat_list, b_mat_list, input_box_list, dt_list)
    zonotopes_quick = get_zonotope_reachset(init_box, a_mat_list, b_mat_list, input_box_list, dt_list, quick=True)

    sim = get_center_sim(init_box, a_mat_list, b_mat_list, input_box_list, dt_list)

    xdim = 0
    ydim = 1
    plt.figure(figsize=(6, 6))
        
    zonotopes[0].plot(col='r-', label='Init', xdim=xdim, ydim=ydim)

    for i, z in enumerate(zonotopes[1:]):
        label = 'Reach Set' if i == 0 else None
        z.plot(label=label, col='k-', xdim=xdim, ydim=ydim)

    for i, z in enumerate(zonotopes_quick[1:]):
        label = 'Reach Set (Quick)' if i == 0 else None
        z.plot(label=label, col='g:', lw=4, xdim=xdim, ydim=ydim)

    plt.plot(sim[:, xdim], sim[:, ydim], 'b-o', label='Center Sim')

    plt.title('Time-Varying with Sim (example_time_varying.py)')
    plt.legend()
    plt.grid()
    plt.savefig(filename)

    # check that the center sim is inside the zonotopes
    for i, (pt, z1, z2) in enumerate(zip(sim, zonotopes, zonotopes_quick)):
        box1 = z1.box_bounds()
        box2 = z2.box_bounds()

        dim = pt_in_box(pt, box1)
        assert dim == -1, f"sim point {pt} dim {dim} is not in zono exact box bounds at step {i}:\n{box1}"

        dim = pt_in_box(pt, box2)
        assert dim == -1, f"sim point {pt} dim {dim} not in zono quick box bounds at step {i}:\n{box2}"

def pt_in_box(pt, box):
    'is the point in the box? if so, return -1, else return outside dim index'

    rv = -1
    tol = 1e-6

    for dim, (x, (lb, ub)) in enumerate(zip(pt, box)):
        if x < lb - tol or x > ub + tol:
            rv = dim
            break

    return rv

if __name__ == "__main__":
    main()
