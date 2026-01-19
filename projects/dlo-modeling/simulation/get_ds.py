import numpy as np
import os

FILENAME = "rope_trajectories.npz"

def main():
    if not os.path.exists(FILENAME):
        print(f"Error: {FILENAME} not found.")
        return

    print(f"Loading {FILENAME}...\n")
    data = np.load(FILENAME, allow_pickle=True)
    
    states = data['states']       # Shape: (N, 10, 70, 3)
    hand_traj = data['hand_traj'] # Shape: (N, 10, 4)
    
    # Check for physics params
    if 'physics_params' in data:
        params = data['physics_params'].item()
        print(f"--- Global Physics Parameters ---")
        print(f"Stiffness: {params.get('stiffness')}")
        print(f"Damping:   {params.get('damping')}")
        print(f"Friction:  {params.get('friction_ground')}")
        print("-" * 40 + "\n")

    print(f"Dataset Shapes:")
    print(f"States:    {states.shape}  (Trajectories, Steps, Links, XYZ)")
    print(f"Hand Traj: {hand_traj.shape}   (Trajectories, Steps, Data[x,y,z,link_id])\n")

    # --- 1. INSPECT HAND TRAJECTORIES ---
    print("=" * 60)
    print(" SAMPLE 1: HAND TRAJECTORIES (First 10 samples)")
    print("=" * 60)
    print("Format: [ Start X,Y,Z (Link ID) ]  --->  [ End X,Y,Z (Link ID) ]")
    
    for i in range(10):
        # Start (step 0) and End (step 9)
        start = hand_traj[i, 0]
        end = hand_traj[i, -1]
        
        # Link ID is at index 3
        link_id_start = int(start[3]) 
        link_id_end = int(end[3])
        
        print(f"Traj {i:02d}: Start [{start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f}] (Link {link_id_start}) "
              f"-> End [{end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f}] (Link {link_id_end})")

    # --- 2. INSPECT ROPE STATES ---
    print("\n" + "=" * 60)
    print(" SAMPLE 2: ROPE STATES (First 10 samples)")
    print("=" * 60)
    print("Showing positions of Link_0 (Start of rope) and Link_69 (End of rope)")
    
    for i in range(10):
        # Initial State (Step 0)
        s0_link0 = states[i, 0, 0]   # Step 0, Link 0
        s0_link69 = states[i, 0, 69] # Step 0, Link 69
        
        # Final Settled State (Step 9)
        s9_link0 = states[i, 9, 0]
        s9_link69 = states[i, 9, 69]
        
        print(f"Traj {i:02d}:")
        print(f"  Start (Step 0):  L0={s0_link0.round(2)} ... L69={s0_link69.round(2)}")
        print(f"  Settle (Step 9): L0={s9_link0.round(2)} ... L69={s9_link69.round(2)}")
        print("-" * 20)

if __name__ == "__main__":
    main()