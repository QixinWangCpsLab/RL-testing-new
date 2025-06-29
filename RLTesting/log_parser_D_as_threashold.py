import re
import ast
import os
import numpy as np
from pathlib import Path
import config_parser

from scipy.stats import linregress



# ---------------------------------Frozen Lake 的解析代码------------------

# 和Frozenlake_Env中定义一致，和训练时保持一致
def mu_state_frozenlake(x1, x2):
    return np.exp(-calc_euclidean_distance_frozenlake(x1, x2))

def mu_action_frozenlake(a1, a2):
    return max(1 - abs(a1 - a2), 0)

# def mu_step_frozenlake(mu_state, mu_action):
#     return mu_state * mu_action

def calc_euclidean_distance_frozenlake(x1, x2):
    x1_d1, x1_d2 = divmod(x1, 4)
    x2_d1, x2_d2 = divmod(x2, 4)
    euclidean_distance = np.sqrt((x1_d1 - x2_d1) ** 2 + (x1_d2 - x2_d2) ** 2)
    return euclidean_distance

def get_x_ref_star_and_a_ref_star_frozenlake(x, pi_ref_i):
    x_ref_star = min(pi_ref_i.keys(), key=lambda s: calc_euclidean_distance_frozenlake(x, s))
    a_ref_star = pi_ref_i[x_ref_star]

    return x_ref_star, a_ref_star


def parse_log_file_fuzzy(log_file_path, D = 1):
    # 读取日志文件内容
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # 使用正则表达式找到rewarded_actions部分
    pi_ref_i_match = re.search(r"rewarded_actions(\{.*?\})", log_content)
    pi_ref_i = ast.literal_eval(pi_ref_i_match.group(1))

    # 使用正则表达式分割内容到每个epoch
    epochs_data = re.split(r"epoch: \d+\n", log_content)

    # 移除第一个元素，它是rewarded_actions之前的内容
    epochs_data.pop(0)

    # 初始化每个epoch的模糊准确率列表
    M_i = []



    for string_epoch_log in epochs_data:
        string_L_i_e = re.findall(r"\(([0-9]+), ([0-9]+)\),", string_epoch_log)
        
        # 公式3.4分子初始化
        sum_mu_step = 0
        # 公式3.5分母初始化
        sum_L_pie_i_e = 0

        for string_x, string_a in string_L_i_e:
            x, a = int(string_x), int(string_a)

            x_ref_star, a_ref_star = get_x_ref_star_and_a_ref_star_frozenlake(x, pi_ref_i)

            mu_state = mu_state_frozenlake(x, x_ref_star)

            mu_action = mu_action_frozenlake(a, a_ref_star)

            # mu_step = mu_step_frozenlake(mu_state, mu_action)
            mu_step = mu_state * mu_action

            # !!!!!这里论文中写的不对
            # if mu_step >= theta_epcp:
            if calc_euclidean_distance_frozenlake(x, x_ref_star) < D:
                sum_mu_step += mu_step
                sum_L_pie_i_e += 1
        
        mu_epoch = sum_mu_step / sum_L_pie_i_e if sum_L_pie_i_e != 0 else 0
        M_i.append(mu_epoch)
    return(M_i)


# -------------------------------------Mountain Car------------------------------
# 和Mountaincar_Env中定义一致，和训练时保持一致
def mu_state_mountaincar(x1, x2):
    distance = abs(x1 - x2)
    # 相似度是距离的递减函数
    # 使用指数递减
    # similarity = np.exp(-distance * 10)
    mu_state = 0
    if distance < 0.5:
        mu_state = 1
    else:
        mu_state = 0
    return mu_state

def mu_action_mountaincar(a1, a2):
    mu_action = 0
    if abs(a1 - a2) < 0.3:
        mu_action = 1
    elif abs(a1 - a2) < 0.5:
        mu_action = 0.7
    else:
        mu_action = 0
    return mu_action


# 这里默认x1为一个数组，数组的第一维表示坐标信息
def calc_distance_mountaincar(x1, x2):
    return abs(x1 - x2)

def get_x_ref_star_and_a_ref_star_mountaincar(x, pi_ref_i):
    x_ref_star = min(pi_ref_i.keys(), key=lambda s: calc_distance_mountaincar(x, s[0]))
    a_ref_star = pi_ref_i[x_ref_star]

    return x_ref_star[0], a_ref_star[0]


def parse_mountaincar_log_file(log_file_path, D = 0.05):
    pi_ref_i = {}

    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # 解析rewarded_actions部分
    pi_ref_i_match = re.search(r"rewarded_actions(\{.*?\})", log_content, re.DOTALL)
    if pi_ref_i_match:
        pi_ref_i_match_str = pi_ref_i_match.group(1)
        pi_ref_i_match_str = re.sub(r", dtype=float32", "", pi_ref_i_match_str)
        pi_ref_i_match_str = re.sub(r"array\((.*?)\)", r"\1", pi_ref_i_match_str)
        pi_ref_i = ast.literal_eval(pi_ref_i_match_str)

    # 解析每个epoch的数据
    epochs_data = re.split(r"epoch: \d+\n", log_content)
    if epochs_data:
        epochs_data.pop(0)  # 移除第一个元素，它是rewarded_actions之前的内容


    M_i = []

    for string_epoch_log in epochs_data:
        pattern = r"array\(\[(.*?),\s+(.*?)\],\s+dtype=float32\),\s+array\(\[(.*?)\],\s+dtype=float32\)"
        matches = re.findall(pattern, string_epoch_log)
        L_i_e = [(tuple(map(float, match[:2])), [float(match[2])]) for match in matches]

        # print(string_L_i_e)

        sum_mu_step = 0
        sum_L_pie_i_e = 0

        for x_and_a_tuple in L_i_e:
            # print(x_and_a_tuple)
            x, a = x_and_a_tuple[0][0], x_and_a_tuple[1][0]
            # print(x, a)

            x_ref_star, a_ref_star = get_x_ref_star_and_a_ref_star_mountaincar(x, pi_ref_i)
            # print(x_ref_star, a_ref_star)

            mu_state = mu_state_mountaincar(x, x_ref_star)

            mu_action = mu_action_mountaincar(a, a_ref_star)

            mu_step = mu_state * mu_action

            if calc_distance_mountaincar(x, x_ref_star) < D:
                sum_mu_step += mu_step
                sum_L_pie_i_e += 1

        mu_epoch = sum_mu_step / sum_L_pie_i_e if sum_L_pie_i_e != 0 else 0
        M_i.append(mu_epoch)


    return M_i










if __name__ == '__main__':
    root_dir = config_parser.parserConfig()['root_dir']
    # path = Path(os.path.join(root_dir, 'RLTesting', 'logs', 'Frozenlake', 'a2c', '[24]', 'time_2024-03-11[24]round_0'))
    # result = parse_log_file_fuzzy(path)


    path = Path(os.path.join(root_dir, 'RLTesting', 'logs', 'Mountaincar', 'a2c', '[26]', 'time_2024-03-21[26]round_2'))
    result = parse_mountaincar_log_file(path)
    slope, _, _, _, _ = linregress(range(len(result)), result)
    
    print(slope)
