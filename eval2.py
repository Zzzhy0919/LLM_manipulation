import argparse
from utils.build_model import get_configs
from softgym.envs.cloth_env import ClothEnv
from utils.visual import action_viz, get_pixel_coord_from_world, get_world_coord_from_pixel
import os
from Policy.demonstrator import Demonstrator
import pickle
import numpy as np
from tqdm import tqdm
from softgym.envs.flex_utils import move_to_pos, rotate_particles
import pyflex
from Policy.agent import Agents
import imageio

def initial_state(random_angle):
    max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
    stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this
    rotate_particles([0, random_angle, 0])
    for _ in range(max_wait_step):
        pyflex.step()
        curr_vel = pyflex.get_velocities()
        if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
            break

def main():
    parser = argparse.ArgumentParser(description="Evaluate")
    # parser.add_argument("--task", type=str, default="SquareTriangle", help="choose task")
    parser.add_argument("--cloth_type", type=str, default="Square", help="choose square cloth or cloth3d cloth")
    parser.add_argument("--gui", action="store_true", help="run with/without gui")
    parser.add_argument("--randomize_pose", action="store_true", help="for squre cloth only")
    parser.add_argument("--model", type=str, help="Evaluate which model")
    parser.add_argument("--agent_model", type=str, help="Evaluate which trained agent model")
    parser.add_argument("--predictor_model", type=str, help="Evaluate which trained predictor model")
    parser.add_argument("--num_eval", type=int, default=100, help="number of eval instances")
    args = parser.parse_args()

    cached_path = os.path.join("configs", args.cloth_type + ".pkl")
    save_root = "eval"
    cloth3d = False if (args.cloth_type == "Square" or args.cloth_type == "Rectangular") else True
    model_config_path = os.path.join("train", "train_configs", args.model + ".yaml")
    configs = get_configs(model_config_path)
    agent_trained_model_path = os.path.join(
        "train",
        "trained_models",
        configs["save_model_name"],
        "model",
        args.agent_model + ".pth",
    )

    agent = Agents[configs["type"]](configs)
    agent.load(agent_trained_model_path)
    env = ClothEnv(
        gui=args.gui,
        cloth3d=cloth3d,
        dump_visualizations=False,
        pick_speed=0.005,
        move_speed=0.005,
        place_speed=0.005,
        lift_height=0.125,
    )
    # load configs
    with open(cached_path, "rb") as f:
        config_data = pickle.load(f)
    cached_configs = config_data["configs"]
    cached_states = config_data["states"]
    print("load {} configs from {}".format(len(cached_configs), cached_path))

    # file
    if args.model == "All":
        task_root = os.path.join(save_root, "Multi")
    else:
        task_root = os.path.join(save_root, "single")
    os.makedirs(task_root, exist_ok=True)
    dirs = os.listdir(task_root)
    if dirs == []:
        max_index = 0
    else:
        existed_index = np.array(dirs).astype(np.int)
        max_index = existed_index.max() + 1

    for i in tqdm(range(args.num_eval)):
        rand_idx = np.random.randint(len(cached_configs))
        config = cached_configs[rand_idx]
        state = cached_states[rand_idx]
        random_angle = np.random.uniform(-40, 40)

        # save file dir
        rgb_folder = os.path.join(task_root, str(max_index + i), "rgb")
        depth_folder = os.path.join(task_root, str(max_index + i), "depth")
        save_folder_viz = os.path.join(task_root, str(max_index + i), "viz")     
        save_folder = os.path.join(task_root, str(max_index + i))
        os.makedirs(rgb_folder, exist_ok=True)
        os.makedirs(depth_folder, exist_ok=True)
        os.makedirs(save_folder_viz, exist_ok=True)
        
        # reset env
        env.reset(config=config, state=state)
        if cloth3d:
            keypoints_index = config_data["keypoints"][rand_idx]
        else:
            keypoints_index = env.get_square_keypoints_idx()
        if args.randomize_pose:
            initial_state(random_angle)

        # initial observation
        action_index = 0
        rgb, depth = env.render_image()
        depth_save = depth.copy() * 255
        depth_save = depth_save.astype(np.uint8)
        imageio.imwrite(os.path.join(depth_folder, str(action_index) + ".png"), depth_save)
        imageio.imwrite(os.path.join(rgb_folder, str(action_index) + ".png"), rgb)
        action_index+=1

        position_pixels=[]
        success_predictions=[]
        rgbs=[]
        instructions=["Pick up the left sleeve of the T-shirt.","Fold it to the right sleeve."]
        for instrucion in instructions:
            position_pixel,success_prediction=agent.get_action(instrucion,depth)
            position_pixels.append(position_pixel)
            success_predictions.append(success_prediction)
        pick_pos=get_world_coord_from_pixel(position_pixels[0],depth, env.camera_params)
        place_pos=get_world_coord_from_pixel(position_pixels[1],depth, env.camera_params)
        env.pick_and_place_single(pick_pos.copy(), place_pos.copy()) #take action

        rgb, depth = env.render_image()
        rgbs.append(rgb)
        depth_save = depth.copy() * 255
        depth_save = depth_save.astype(np.uint8)
        imageio.imwrite(os.path.join(rgb_folder, str(action_index) + ".png"), rgb)
        imageio.imwrite(os.path.join(depth_folder, str(action_index) + ".png"), depth_save)

        img = action_viz(rgbs[0], position_pixels[0], position_pixels[1])
        imageio.imwrite(os.path.join(save_folder_viz, str(0) + ".png"), img)



if __name__ == "__main__":
    main()