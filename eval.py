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
import openai
import time

# openai.api_key = "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" #blueYu
# openai.api_key = "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" #tsinghua
openai.api_base="https://ai98.vip/v1"
openai.api_key = "sk-CcCvez7msME97Kwp5cDb5947E8D7466a94Cf400b21285d0d" #taobao
class Chat:
    def __init__(self):
        self._base_prompt = None
        self._context = None
    
    # 打印对话
    def show_conversation(self,msg_list):
        for msg in msg_list:
            if msg['role'] == 'user':
                print(f"\U0001f47b: {msg['content']}\n")
            else:
                print(f"\U0001f47D: {msg['content']}\n")

    # 提示chatgpt
    def ask(self, prompt):
        instructions = '''
            “here are the instruction-templates:(in "[]")
            [
            "Fold the Trousers in half, starting from the {which1} and ending at the {which2}.",
            "Fold the Trousers, {which1} side over {which2} side.",
            "Bend the Trousers in half, from {which1} to {which2}.",
            "Crease the Trousers down the middle, from {which1} to {which2}.",
            "Fold the Trousers in half horizontally, {which1} to {which2}.",
            "Make a fold in the Trousers, starting from the {which1} and ending at the {which2}.",
            "Fold the Trousers in half, aligning the {which1} and {which2} sides.",
            "Fold the Trousers, orientating from the {which1} towards the {which2}.",
            "Fold the Trousers in half, with the {which1} side overlapping the {which2}.",
            "Create a fold in the Trousers, going from {which1} to {which2}."
            "Bring the {which1} side of the Trousers towards the {which2} side and fold them in half.",
            "Fold the waistband of the Trousers in half, from {which1} to {which2}.",
            "Fold the Trousers neatly, from the {which1} side to the {which2} side.",

            "Fold the Trousers in half vertically from top to bottom.",
            "Create a fold in the Trousers from the waistband to the hem.",
            "Fold the Trousers along the vertical axis, starting from the top.",
            "Fold the Trousers in half lengthwise, beginning at the waistband.",
            "Fold the Trousers vertically, starting at the waistband.",
            "Fold the Trousers in half, starting from the top edge.",
            "Fold the Trousers by bringing the waistband down to meet the hem.",
            "Fold the Trousers in half vertically, starting at the upper edge.",
            "Fold the Trousers by bringing the waistband down to meet the bottom.",
            "Fold the Trousers in half, starting from the top seam.",
            "Fold the Trousers in half, bringing the top towards the hem.",

            "Bring the {which1} side of the Trousers towards the {which2} side and fold them in half.",
            "Fold the waistband of the Trousers in half, from {which1} to {which2}.",
            "Fold the Trousers neatly, from the {which1} side to the {which2} side.",
            "Fold the Trousers, making a crease from the {which1} to the {which2}.",
            ]
            Then there are some additional notes :
            1.{which1} and {which2} can be random in: location_list=['left' ,'right', 'top' ,'bottom' ,'waistband','hem']
            2.you can replace words in location_list with other words in location_list.
            ”
            '''
        instructions2 = '''
            I will give you some widely instrucion,and you should reply me three sentences following the instruction-templates I've gave you.
            For example,
            users:"fold the Trousers to minimum"
            assistant:
            1."Create a fold in the Trousers from the waistband to the hem"
            2."Bring the right side of the Trousers towards the left side and fold them in half."
            3."Fold the Trousers in half vertically, beginning at the top."
            ""


            '''
        assistant1 = f'Got it. I will complete what you give me next.'
        ret = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that pays attention to the user's top-level instructions and writes low-level instruction sentences according to the instruction-templates I will give you."},
                {"role": "user", "content": instructions},
                {"role": "assistant", "content": assistant1},
                {"role": "user", "content": instructions2},
                {"role": "user", "content": prompt}
            ]
            
        )
        answer = ret.choices[0].message.content
        return answer

#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
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
    parser.add_argument("--task", type=str, default="SquareTriangle", help="choose task")
    parser.add_argument("--cloth_type", type=str, default="Square", help="choose square cloth or cloth3d cloth")
    parser.add_argument("--gui", action="store_true", help="run with/without gui")
    parser.add_argument("--randomize_pose", action="store_true", help="for squre cloth only")
    parser.add_argument("--model", type=str, help="Evaluate which model")
    parser.add_argument("--agent_model", type=str, help="Evaluate which trained agent model")
    parser.add_argument("--predictor_model", type=str, help="Evaluate which trained predictor model")
    parser.add_argument("--num_eval", type=int, default=5, help="number of eval instances")
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
    predictor_trained_model_path = os.path.join(
        "train",
        "trained_models",
        configs["save_model_name"],
        "predictor",
        args.predictor_model + ".pth",
    )
    task = args.task

    # set task, env & agent
    demonstrator = Demonstrator[task]()
    agent = Agents[configs["type"]](configs)
    agent.load(agent_trained_model_path, predictor_trained_model_path)
    env = ClothEnv(
        gui=args.gui,
        cloth3d=cloth3d,
        dump_visualizations=False,
        pick_speed=demonstrator.pick_speed,
        move_speed=demonstrator.move_speed,
        place_speed=demonstrator.place_speed,
        lift_height=demonstrator.lift_height,
    )
    # load configs
    with open(cached_path, "rb") as f:
        config_data = pickle.load(f)
    cached_configs = config_data["configs"]
    cached_states = config_data["states"]
    print("load {} configs from {}".format(len(cached_configs), cached_path))

    # file
    if args.model == "All":
        task_root = os.path.join(save_root, "Multi", task)
    else:
        task_root = os.path.join(save_root, "single", task)
    os.makedirs(task_root, exist_ok=True)
    dirs = os.listdir(task_root)
    if dirs == []:
        max_index = 0
    else:
        existed_index = np.array(dirs).astype(int)
        max_index = existed_index.max() + 1

    for i in tqdm(range(args.num_eval)):
        rand_idx = np.random.randint(len(cached_configs))
        config = cached_configs[rand_idx]
        state = cached_states[rand_idx]
           
        if task == "StraightFold":
            #fix simulation bugs of square
            random_angle = np.random.uniform(-80, 80)
        elif cloth3d:           
            random_angle = np.random.uniform(-40, 40)
        else:
            random_angle = np.random.uniform(0, 40)

        
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
        
        # oracle excuted
        if task == "StraightFold":
            eval_seen_instructions,  eval_unseen_instructions, eval_unseen_tasks = demonstrator.get_eval_instruction(random_angle)
        else: 
            eval_seen_instructions,  eval_unseen_instructions, eval_unseen_tasks = demonstrator.get_eval_instruction() 
        eval_datas = [eval_seen_instructions, eval_unseen_instructions,eval_unseen_tasks]
        eval_name_list = ["si","usi","ut"]

        eval_results =  {
        "si":[],
        "usi":[],
        "ut":[],
        } 

        eval_flags =  {
        "si":[],
        "usi":[],
        "ut":[],
        } 
               
        for eval_index in range(1):  #eval seen instructions, unseen instructions, unseent task
            print("start...")
            cha = Chat()
            query = input("please input your instruction:")
            ans = cha.ask(query)
            lines = ans.strip().split("\n")
            # 然后去除每行的数字序号和空格
            my_list = [line.split('. ', 1)[1].strip('"') for line in lines if line]
            print(my_list)
            eval_data = eval_datas[eval_index]
            eval_name = eval_name_list[eval_index]
            print("eval_stage:", eval_name)
            pick_idxs = eval_data["pick"]
            place_idxs = eval_data["place"]
            gammas = eval_data["gammas"]
            unseen_flags = eval_data["flags"]
            instructions = my_list
            oracle_results = []
            model_results = []
                                                   
            # reset env
            env.reset(config=config, state=state)
            if args.randomize_pose:
                initial_state(random_angle)
            #oracle
            action_index = 0      
            for pick_idx, place_idx, gamma in zip(pick_idxs, place_idxs, gammas):      
                keypoints_pos = env.get_keypoints(keypoints_index)
                pick_pos = keypoints_pos[pick_idx]
                place_pos = keypoints_pos[place_idx]
                place_pos = pick_pos + gamma * (place_pos - pick_pos)
                env.pick_and_place_single(pick_pos.copy(), place_pos.copy()) 
                action_index += 1
                # save
                rgb, _ = env.render_image()
                imageio.imwrite(os.path.join(rgb_folder, eval_name + "_ori_" + str(action_index) + ".png"), rgb)
                particle_pos = pyflex.get_positions().reshape(-1,4)[:,:3]
                oracle_results.append(particle_pos)

            # reset env
            env.reset(config=config, state=state)
            if args.randomize_pose:
                initial_state(random_angle)

            # model excuted 
            pick_pixels = []
            place_pixels = []
            rgbs = []

            # initial observation
            action_index = 0
            rgb, depth = env.render_image()
            depth_save = depth.copy() * 255
            depth_save = depth_save.astype(np.uint8)
            imageio.imwrite(os.path.join(depth_folder, eval_name+"_"+str(action_index) + ".png"), depth_save)
            imageio.imwrite(os.path.join(rgb_folder, eval_name+"_"+str(action_index) + ".png"), rgb)
            rgbs.append(rgb)

            # test begin
            for pick_idx, place_idx, gamma, instruction, unseen_flag in zip(pick_idxs, place_idxs, gammas, instructions, unseen_flags):     
                print("task: " + instruction)
                if eval_index < 4: # eval seen instructions, unseen instructions,
                    if unseen_flag == 1: #oracle execute action
                        keypoints_pos = env.get_keypoints(keypoints_index)
                        pick_pos = keypoints_pos[pick_idx]
                        place_pos = keypoints_pos[place_idx]
                        place_pos = pick_pos + gamma * (place_pos - pick_pos)
                        pick_pixel = get_pixel_coord_from_world(pick_pos, depth.shape, env.camera_params)
                        place_pixel = get_pixel_coord_from_world(place_pos, depth.shape, env.camera_params)
                    else: #model execute action
                        pick_pixel, place_pixel, success_prediction = agent.get_action(instruction, depth)
                        pick_pos = get_world_coord_from_pixel(pick_pixel, depth, env.camera_params)
                        place_pos = get_world_coord_from_pixel(place_pixel, depth, env.camera_params)
                # else: #eval unseen tasks
                #     if unseen_flag == 0: #oracle execute action
                #         keypoints_pos = env.get_keypoints(keypoints_index)
                #         pick_pos = keypoints_pos[pick_idx]
                #         place_pos = keypoints_pos[place_idx]
                #         place_pos = pick_pos + gamma * (place_pos - pick_pos)
                #         pick_pixel = get_pixel_coord_from_world(pick_pos, depth.shape, env.camera_params)
                #         place_pixel = get_pixel_coord_from_world(place_pos, depth.shape, env.camera_params)
                #     else: #model execute action
                #         pick_pixel, place_pixel, success_prediction = agent.get_action(instruction, depth)
                #         pick_pos = get_world_coord_from_pixel(pick_pixel, depth, env.camera_params)
                #         place_pos = get_world_coord_from_pixel(place_pixel, depth, env.camera_params)

                env.pick_and_place_single(pick_pos.copy(), place_pos.copy()) #take action

                # render & update frames & save
                action_index += 1
                rgb, depth = env.render_image()
                depth_save = depth.copy() * 255
                depth_save = depth_save.astype(np.uint8)
                imageio.imwrite(os.path.join(rgb_folder, eval_name+"_"+str(action_index) + ".png"), rgb)
                imageio.imwrite(os.path.join(depth_folder, eval_name+"_"+str(action_index) + ".png"), depth_save)
                rgbs.append(rgb)
                pick_pixels.append(pick_pixel)
                place_pixels.append(place_pixel)
                particle_pos = pyflex.get_positions().reshape(-1,4)[:,:3]
                model_results.append(particle_pos) 
        
            #record results
            eval_results[eval_name]= [oracle_results,model_results]
            eval_flags[eval_name]= unseen_flags

            # action viz
            num_actions = len(pick_pixels)
            for act in range(num_actions + 1):
                if act < num_actions:
                    img = action_viz(rgbs[act], pick_pixels[act], place_pixels[act])
                else:
                    img = rgbs[act]
                imageio.imwrite(os.path.join(save_folder_viz, eval_name+"_"+str(act) + ".png"), img)
        
        #save results
        with open(os.path.join(save_folder, "resu.pkl"), "wb+") as f:
            data = {"eval_results": eval_results, 
                    "eval_flags": eval_flags}
            pickle.dump(data, f)



if __name__ == "__main__":
    main()
