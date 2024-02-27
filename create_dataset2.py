import os
import pickle
import imageio
import numpy as np
import argparse

Done = np.array([0, 0])

def create_dataset(root, tasks, save_path, with_done, use_rgb, n_demos):
    depths = []
    positions=[]
    instructions = []
    success = []
    rgbs = []
    each_task_num = 0
    
    if "All" in tasks:
        tasks = os.listdir(root)
        print("Load All Tasks: ", tasks)

    for task in tasks:
        each_task_num = 0
        task_path = os.path.join(root, task)
        trajs = os.listdir(task_path)
        for traj in trajs:
            if each_task_num >= n_demos:
                break 
            traj_path = os.path.join(task_path, traj)
            with open(os.path.join(traj_path, "info.pkl"), "rb") as f:
                data = pickle.load(f)
                langs = data["instruction"]
                prims = data["primitive"]
                if task=="Pick_Tshirt":
                    pick_pixels=data["pick"]
                elif task=="Place_Tshirt":
                    place_pixels=data["place"]
            each_task_num += 1
            depth_path = os.path.join(task_path, traj, "depth")
            rgb_path = os.path.join(task_path, traj, "rgb")

            if not with_done:
                if task=="Pick_Tshirt":
                    # insert actions & instructions
                    positions.append(pick_pixels[0])
                    instructions.append(langs[0])
                    success.append(0)
                elif task=="Place_Tshirt":
                    # insert actions & instructions
                    positions.append(place_pixels[0])
                    instructions.append(langs[0])
                    success.append(0)
                    # observations
                depths.append(imageio.imread(os.path.join(depth_path, str(0) + ".png")))
                if use_rgb:
                    rgbs.append(imageio.imread(os.path.join(rgb_path, str(0) + ".png")))
            else:
                if task=="Pick_Tshirt":
                    positions.append(pick_pixels[0])
                    instructions.append(langs[0])
                    success.append(0)
                                
                    positions.append(Done)
                    instructions.append(langs[0])
                    success.append(1)
                elif task=="Place_Tshirt":
                    positions.append(place_pixels[0])
                    instructions.append(langs[0])
                    success.append(0)
                                
                    positions.append(Done)
                    instructions.append(langs[0])
                    success.append(1)
                            
                # observations
                depths.append(imageio.imread(os.path.join(depth_path, str(0) + ".png")))
                depths.append(imageio.imread(os.path.join(depth_path, str(1) + ".png")))
                if use_rgb:
                    rgbs.append(imageio.imread(os.path.join(rgb_path, str(0) + ".png")))
                    rgbs.append(imageio.imread(os.path.join(rgb_path, str(1) + ".png")))
    assert len(depths)==len(positions)==len(instructions)==len(success)
    #print(positions,instructions)
    dataset = {
        "depth": depths,
        "position": positions,
        "instruction": instructions,
        "success": success,
    }


    if use_rgb:
        dataset.update({"rgbs": rgbs})
    with open(save_path, "wb+") as f:
        pickle.dump(dataset, f)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("--tasks", type=str, help="choose single task / all task(all)")
    parser.add_argument("--with_done", action="store_true", help="choose with_success or not")
    parser.add_argument("--use_rgb", action="store_true", help="choose with inst feature or not")
    parser.add_argument("--root", type=str, default="./raw_data")
    parser.add_argument("--save_path_root", type=str, default="./data")
    parser.add_argument("--n_demos", type=int, default=100, help="num of demos")
    args = parser.parse_args()
    
    if args.tasks == "All":
        if args.with_done:
            save_path = os.path.join(args.save_path_root, "All_done_100.pkl")
        else:
            save_path = os.path.join(args.save_path_root, "All_100.pkl")
    else:
        if args.with_done:
            save_path = os.path.join(args.save_path_root, args.tasks+"_done"+".pkl")
        else:
            save_path = os.path.join(args.save_path_root, args.tasks+".pkl")
 
    tasks = [args.tasks]
    create_dataset(args.root, tasks , save_path, args.with_done, args.use_rgb, args.n_demos)
