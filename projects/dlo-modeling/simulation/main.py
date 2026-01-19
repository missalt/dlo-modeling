import numpy as np
import mujoco
import time
import os
import concurrent.futures
import multiprocessing
import threading

# --- Configuration ---
XML_PATH = "rope_chain.xml"
OUTPUT_FILENAME = "rope_trajectories.npz"

# Process Configuration
TOTAL_PROCESSES = 30
TRAJECTORIES_PER_WORKER = 1000  
RECORDING_STEPS = 10          # 1 Start + 8 Move + 1 End

# Trajectory Settings
MOVE_DURATION = 3.0          
LIFT_DELTA_Z = 0.01          
TRANSPORT_RADIUS = 0.25      
MIN_SAFE_Z = 0.02            

# Physics Settings
TIMESTEP = 0.002             
INITIAL_SETTLE_STEPS = 3000   # Long settle as requested
BETWEEN_SETTLE_STEPS = 100    
POST_DROP_SETTLE_TIME = 0.5  
POST_DROP_STEPS = int(POST_DROP_SETTLE_TIME / TIMESTEP)

# Fixed Physics Constants
BASE_DAMPING = 0.05           
BASE_GROUND_FRICTION = 1.0    

def get_ids(model):
    link_ids = []
    weld_ids = []
    i = 0
    while True:
        link_name = f"link_{i}"
        weld_name = f"weld_{i}"
        lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
        if lid == -1: break
        link_ids.append(lid)
        weld_ids.append(wid)
        i += 1
    if not link_ids: raise RuntimeError("No links found.")
    return np.array(link_ids, dtype=int), np.array(weld_ids, dtype=int)

def smooth_interp(p_start, p_end, steps):
    t = np.linspace(0, 1, steps)
    s = 0.5 * (1 - np.cos(t * np.pi))
    return np.outer(1 - s, p_start) + np.outer(s, p_end)

def generate_low_lift_trajectory(start_pos, total_steps):
    p0 = start_pos
    p1 = p0 + np.array([0, 0, LIFT_DELTA_Z])
    
    theta = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(0.05, TRANSPORT_RADIUS)
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    p2 = p1 + np.array([dx, dy, 0])
    
    target_z = max(MIN_SAFE_Z, p0[2]) 
    p3 = np.array([p2[0], p2[1], target_z])
    
    steps_phase = total_steps // 3
    remainder = total_steps - (steps_phase * 3)
    
    traj_1 = smooth_interp(p0, p1, steps_phase)
    traj_2 = smooth_interp(p1, p2, steps_phase)
    traj_3 = smooth_interp(p2, p3, steps_phase + remainder)
    
    return np.vstack([traj_1, traj_2, traj_3]), p3 - p0

def get_worker_config(worker_id):
    """
    Assigns physics parameters based on worker ID.
    Total: 30 Workers.
    """
    
    # Group A: Workers 0-24 (25 Workers) -> Vary Stiffness
    # Log-space gives more values near 0.001-0.01 and fewer near 0.5
    stiff_range = np.geomspace(0.001, 0.5, 25)
    
    # Group B: Workers 25-29 (5 Workers) -> Vary Friction (0.5 or 1.0)
    # Alternating pattern: 0.5, 1.0, 0.5, 1.0, 0.5
    fric_pattern = [0.5, 1.0, 0.5, 1.0, 0.5]

    if worker_id < 25:
        # Vary Stiffness, Fixed Friction (0.8)
        config = {
            'stiffness': stiff_range[worker_id],
            'rope_friction': 0.8,
            'damping': BASE_DAMPING,
            'ground_friction': BASE_GROUND_FRICTION
        }
    else:
        # Fixed Stiffness (0.005), Vary Friction
        idx = worker_id - 25
        config = {
            'stiffness': 0.005,
            'rope_friction': fric_pattern[idx],
            'damping': BASE_DAMPING,
            'ground_friction': BASE_GROUND_FRICTION
        }
        
    return config

def worker_routine(worker_id, num_trajectories, global_counter, counter_lock):
    np.random.seed(worker_id * int(time.time()) % 123456789)
    
    # 1. Setup Model
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    m.opt.timestep = TIMESTEP
    
    # 2. Apply Physics
    config = get_worker_config(worker_id)
    
    for i in range(1, 70):
        j_name = f"j_{i}"
        j_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id != -1:
            m.jnt_stiffness[j_id] = config['stiffness']
            dof_adr = m.jnt_dofadr[j_id]
            m.dof_damping[dof_adr:dof_adr+3] = config['damping']

    g_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if g_id != -1:
        m.geom_friction[g_id, 0] = config['ground_friction']
        
    for g_id in range(m.ngeom):
        g_type = m.geom_type[g_id]
        if g_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            m.geom_friction[g_id, 0] = config['rope_friction']

    # 3. Storage
    link_ids, weld_ids = get_ids(m)
    num_links = len(link_ids)
    
    dataset_states = np.zeros((num_trajectories, RECORDING_STEPS, num_links, 3), dtype=np.float32)
    dataset_hand   = np.zeros((num_trajectories, RECORDING_STEPS, 4), dtype=np.float32)
    dataset_params = np.zeros((num_trajectories, 4), dtype=np.float32)
    
    sim_steps_total = int(MOVE_DURATION / TIMESTEP)
    intermediate_indices = np.linspace(0, sim_steps_total - 1, 8, dtype=int)

    # 4. Loop
    mujoco.mj_resetData(m, d)
    d.eq_active[:] = 0
    
    # Initial Long Settle
    for _ in range(INITIAL_SETTLE_STEPS):
        mujoco.mj_step(m, d)

    for t_idx in range(num_trajectories):
        dataset_params[t_idx] = [
            config['stiffness'], 
            config['rope_friction'], 
            config['damping'], 
            config['ground_friction']
        ]
        
        rec_count = 0
        
        # Grab
        target_idx = np.random.randint(0, num_links)
        lid = link_ids[target_idx]
        wid = weld_ids[target_idx]

        d.mocap_pos[0] = d.xpos[lid]
        d.mocap_quat[0] = d.xquat[lid]
        mujoco.mj_step(m, d) 
        
        m.eq_solref[wid] = [0.01, 1.0] 
        d.eq_active[wid] = 1

        # Plan
        start_pos = d.mocap_pos[0].copy()
        traj_points, _ = generate_low_lift_trajectory(start_pos, sim_steps_total)
        
        # STEP 0: Record Start
        dataset_states[t_idx, rec_count] = d.xpos[link_ids].astype(np.float32)
        h_pos = d.mocap_pos[0].astype(np.float32)
        dataset_hand[t_idx, rec_count] = [h_pos[0], h_pos[1], h_pos[2], float(target_idx)]
        rec_count += 1

        # STEPS 1-8: Move & Record
        for step in range(sim_steps_total):
            d.mocap_pos[0] = traj_points[step]
            mujoco.mj_step(m, d)
            
            if step in intermediate_indices:
                if rec_count < RECORDING_STEPS - 1:
                    dataset_states[t_idx, rec_count] = d.xpos[link_ids].astype(np.float32)
                    h_pos = d.mocap_pos[0].astype(np.float32)
                    dataset_hand[t_idx, rec_count] = [h_pos[0], h_pos[1], h_pos[2], float(target_idx)]
                    rec_count += 1

        # Release
        d.eq_active[wid] = 0 
        d.mocap_pos[0] = [0, 0, 2.0]
        
        # Post-Drop Settle
        for _ in range(POST_DROP_STEPS):
            mujoco.mj_step(m, d)

        # STEP 9: Record Settled
        dataset_states[t_idx, RECORDING_STEPS - 1] = d.xpos[link_ids].astype(np.float32)
        dataset_hand[t_idx, RECORDING_STEPS - 1] = [0.0, 0.0, 2.0, float(target_idx)]

        # Global Counter
        with counter_lock:
            global_counter.value += 1
            
        # Between Episode Settle
        for _ in range(BETWEEN_SETTLE_STEPS):
            mujoco.mj_step(m, d)

    partial_filename = f"partial_data_{worker_id}.npz"
    np.savez_compressed(partial_filename, states=dataset_states, hand=dataset_hand, configs=dataset_params)
    return partial_filename

def monitor_throughput(global_counter, counter_lock, stop_event, total_target):
    print("\n--- Waiting for Initial Settle (3000 steps)... ---")
    start_time = None
    
    while not stop_event.is_set():
        with counter_lock:
            val = global_counter.value
        
        if val > 0:
            start_time = time.time()
            print("--- First trajectory finished. Tracking Global Speed. ---")
            break
        time.sleep(0.1)

    if start_time is None: return

    while not stop_event.is_set():
        time.sleep(2.0)
        current_time = time.time()
        with counter_lock:
            current_count = global_counter.value
            
        elapsed = current_time - start_time
        if elapsed > 1.0:
            tps = current_count / elapsed
            print(f"Progress: {current_count}/{total_target} | Global Trajectories/Sec: {tps:.2f}")
        
        if current_count >= total_target:
            break

def main():
    if not os.path.exists(XML_PATH):
        print(f"Error: {XML_PATH} not found.")
        return

    total_target = TOTAL_PROCESSES * TRAJECTORIES_PER_WORKER
    print(f"Starting Collection:")
    print(f"  Processes: {TOTAL_PROCESSES}")
    print(f"  Total Target: {total_target} trajectories")

    manager = multiprocessing.Manager()
    global_counter = manager.Value('i', 0)
    counter_lock = manager.Lock()
    stop_event = threading.Event()

    monitor_thread = threading.Thread(
        target=monitor_throughput, 
        args=(global_counter, counter_lock, stop_event, total_target)
    )
    monitor_thread.start()

    start_time_main = time.time()
    temp_files = []

    print("Launching Process Pool...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=TOTAL_PROCESSES) as executor:
        futures = []
        for i in range(TOTAL_PROCESSES):
            futures.append(executor.submit(
                worker_routine, i, TRAJECTORIES_PER_WORKER, global_counter, counter_lock
            ))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                fname = future.result()
                temp_files.append(fname)
            except Exception as e:
                print(f"Worker Exception: {e}")

    stop_event.set()
    monitor_thread.join()

    print(f"\nMerging files...")
    all_states, all_hands, all_configs = [], [], []
    
    for fname in temp_files:
        if os.path.exists(fname):
            with np.load(fname) as d:
                all_states.append(d['states'])
                all_hands.append(d['hand'])
                all_configs.append(d['configs'])
            os.remove(fname)

    if all_states:
        final_states = np.concatenate(all_states, axis=0)
        final_hands = np.concatenate(all_hands, axis=0)
        final_configs = np.concatenate(all_configs, axis=0)
        
        print(f"Saving to {OUTPUT_FILENAME}")
        print(f"States:  {final_states.shape}")
        print(f"Configs: {final_configs.shape} (stiff, fric_rope, damp, fric_ground)")
        
        np.savez_compressed(
            OUTPUT_FILENAME, 
            states=final_states, 
            hand_traj=final_hands,
            configs=final_configs
        )
        print(f"Total Time: {time.time() - start_time_main:.2f}s")
    else:
        print("No data collected.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()