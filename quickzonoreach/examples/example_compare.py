'''
quickzonoreach compare example with quick vs no quick

Stanley Bak, Feb 2020
'''

import matplotlib.pyplot as plt

from example_profile import run_single_profile

def main():
    'make comparison plot'

    filename = 'compare.png'
    print(f"making ({filename}) with quick=True vs quick=False (4 secs)...")
    
    dims = 8
    num_steps = 128

    plt.figure(figsize=(6, 3.5))

    quick_list = [False, True]
    col_list = ['b-', 'g:']
    label_list = ['Exact', 'Quick']
    lw_list = [1, 4]
    
    for quick, col, label, lw in zip(quick_list, col_list, label_list, lw_list):
        z = run_single_profile(dims, num_steps, quick, save_all=False)
        z.plot(col, lw=lw, label=label)

    # plot init set in red
    pts = [(-5, 0), (-4, 0), (-4, 1), (-5, 1), (-5, 0)]
    plt.plot(*zip(*pts), 'r-', label='Init')

    plt.title(f'Accuracy Demo ({dims} dims, {num_steps} steps, example_compare.py)')
    plt.legend()
    plt.grid()
    plt.savefig(filename)

if __name__ == '__main__':
    main()
