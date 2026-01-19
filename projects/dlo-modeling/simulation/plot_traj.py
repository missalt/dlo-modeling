import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

FILENAME = "rope_trajectories.npz"
PLOT_DIR = "plots/frames"

def main():
    if not os.path.exists(FILENAME):
        print(f"Error: {FILENAME} not found.")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    print(f"Loading {FILENAME}...")
    data = np.load(FILENAME, allow_pickle=True)
    
    states = data['states']       # (N, 10, 70, 3)
    hand_traj = data['hand_traj'] # (N, 10, 4)
    configs = data['configs']     # (N, 4)
    
    num_samples = 5
    total_traj = states.shape[0]
    indices = np.random.choice(total_traj, num_samples, replace=False)
    
    print(f"Generating plots for {num_samples} trajectories...")

    for idx in indices:
        # Changed: Removed constrained_layout=True to fix the collapse warning
        fig = plt.figure(figsize=(24, 10))
        
        rope_t = states[idx]
        hand_t = hand_traj[idx]
        params = configs[idx]
        
        # --- 1. Calculate Bounds ---
        start_hand = hand_t[0, :3]
        end_hand = hand_t[8, :3]
        move_vec = end_hand - start_hand
        
        all_points = rope_t.reshape(-1, 3)
        
        min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
        min_y, max_y = all_points[:, 1].min(), all_points[:, 1].max()
        max_z = all_points[:, 2].max()
        
        # Center the plot
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        
        # Ensure a minimum field of view of 40cm so we don't zoom in too much on a tiny rope
        range_x = max_x - min_x
        range_y = max_y - min_y
        max_range = max(range_x, range_y, 0.4) / 2.0
        
        xlims = [mid_x - max_range, mid_x + max_range]
        ylims = [mid_y - max_range, mid_y + max_range]
        # Z from slightly below zero (to show floor contact) to max height + padding
        zlims = [-0.01, max(max_z + 0.1, 0.3)] 

        # --- 2. Title ---
        action_text = f"Action: [dx={move_vec[0]:.2f}, dy={move_vec[1]:.2f}]"
        param_text = f"Stiff: {params[0]:.4f} | Rope Fric: {params[1]:.1f}"
        fig.suptitle(f"Trajectory {idx} | {action_text} | {param_text}", fontsize=18)

        # --- 3. Plot 10 Steps ---
        for step in range(10):
            ax = fig.add_subplot(2, 5, step + 1, projection='3d')
            
            if step == 0:
                color, label = 'blue', "Start (0)"
            elif step == 9:
                color, label = 'red', "Settled (9)"
            else:
                color, label = 'teal', f"Move ({step})"

            # Plot Rope
            ax.plot(rope_t[step, :, 0], rope_t[step, :, 1], rope_t[step, :, 2], 
                    color=color, linewidth=2, marker='o', markersize=2, alpha=0.9)

            # Plot Hand
            if step < 9:
                hx, hy, hz = hand_t[step, :3]
                ax.scatter(hx, hy, hz, color='black', s=80, marker='X')
                
                # Ghost Arrow
                ax.quiver(start_hand[0], start_hand[1], start_hand[2], 
                          move_vec[0], move_vec[1], move_vec[2], 
                          color='magenta', alpha=0.2, arrow_length_ratio=0.15, linewidth=2)

            # Enforce Limits (Normalization)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_zlim(zlims)
            ax.set_title(label, fontsize=10)
            
            # Hide Ticks
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            
            # View Angle
            ax.view_init(elev=40, azim=-45)

        # Use tight_layout instead of constrained_layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.90) # Leave room for title
        
        save_path = os.path.join(PLOT_DIR, f"traj_{idx}_normalized.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"Saved {save_path}")

    print("Done.")

if __name__ == "__main__":
    main()