import numpy as np
import pickle
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import argparse

def make_gif(traj, fname, title=''):
    fig, ax = plt.subplots(figsize=(2, 2))

    def update(i):
        if i % 20 == 0:
            print(i)
        im_normed = traj[i]
        ax.imshow(im_normed)
        ax.set_title(title, fontsize=20)
        ax.set_axis_off()

    anim = FuncAnimation(fig, update, frames=np.arange(0, min(1000, len(traj))), interval=50)
    anim.save(fname, dpi=80, writer='imagemagick')
    plt.close()
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--trajectories_file', default='trajectories.pkl')
    parser.add_argument('--output_file', default='trajectories.gif')
    parser.add_argument('--title', default='')
    
    args = parser.parse_args()
    
    ts = pickle.load(open(args.trajectories_file, 'rb'))
    
    make_gif(ts[0]['observations'], args.output_file, args.title)