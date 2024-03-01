import re
import ast
import os
import numpy as np


def parse_log_file(log_file_path):
    # 读取日志文件内容
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # 使用正则表达式找到rewarded_actions部分
    rewarded_actions_match = re.search(r"rewarded_actions(\{.*?\})", log_content)
    rewarded_actions = ast.literal_eval(rewarded_actions_match.group(1))

    # 使用正则表达式分割内容到每个epoch
    epochs_data = re.split(r"epoch: \d+\n", log_content)

    # 移除第一个元素，它是rewarded_actions之前的内容
    epochs_data.pop(0)

    # 初始化每个epoch的准确率列表
    epoch_accuracies = []

    # 分析每个epoch
    for epoch_data in epochs_data:
        # 使用正则表达式找到所有的状态和动作对
        actions_data = re.findall(r"\[([0-9]+)\],\[([0-9]+)\]", epoch_data)

        # 初始化计数器
        total_actions = len(actions_data)
        rewarded_count = 0

        # 检查每个动作是否在rewarded_actions中
        for state, action in actions_data:
            state, action = int(state), int(action)
            # 检查动作是否是对应状态的rewarded action
            if rewarded_actions.get(state) == action:
                rewarded_count += 1

        # 计算准确率并添加到列表
        accuracy = rewarded_count / total_actions if total_actions > 0 else 0
        epoch_accuracies.append(accuracy)

    return epoch_accuracies



def parse_log_file_new(log_file_path):
    # 读取日志文件内容
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # 使用正则表达式找到rewarded_actions部分
    rewarded_actions_match = re.search(r"rewarded_actions(\{.*?\})", log_content)
    rewarded_actions = ast.literal_eval(rewarded_actions_match.group(1))

    # 使用正则表达式分割内容到每个epoch
    epochs_data = re.split(r"epoch: \d+\n", log_content)

    # 移除第一个元素，它是rewarded_actions之前的内容
    epochs_data.pop(0)

    # 初始化每个epoch的准确率列表
    epoch_accuracies = []

    # 分析每个epoch
    for epoch_data in epochs_data:
        # 使用正则表达式找到所有的状态和动作对
        # (0, 0),
        actions_data = re.findall(r"\(([0-9]+), ([0-9]+)\),", epoch_data)

        # 初始化计数器
        total_actions = len(actions_data)
        rewarded_count = 0

        # 检查每个动作是否在rewarded_actions中
        for state, action in actions_data:
            state, action = int(state), int(action)
            # 检查动作是否是对应状态的rewarded action
            if rewarded_actions.get(state) == action:
                rewarded_count += 1

        # 计算准确率并添加到列表
        accuracy = rewarded_count / total_actions if total_actions > 0 else 0
        epoch_accuracies.append(accuracy)

    return epoch_accuracies


def calculate_reward_membership(action, ideal_action, fuzziness=0.99):
    """
    计算一个动作相对于理想动作的模糊隶属度。
    Args:
        action: 实际采取的动作
        ideal_action: 理想的动作（被认为是rewarded的动作）
        fuzziness: 模糊度参数，定义了非理想动作的惩罚程度

    Returns:
        float: 表示隶属度的数值
    """
    if action == ideal_action:
        return 1
    else:
        return 0



def parse_log_file_fuzzy(log_file_path):
    # 读取日志文件内容
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # 使用正则表达式找到rewarded_actions部分
    rewarded_actions_match = re.search(r"rewarded_actions(\{.*?\})", log_content)
    rewarded_actions = ast.literal_eval(rewarded_actions_match.group(1))

    # 使用正则表达式分割内容到每个epoch
    epochs_data = re.split(r"epoch: \d+\n", log_content)

    # 移除第一个元素，它是rewarded_actions之前的内容
    epochs_data.pop(0)

    # 初始化每个epoch的模糊准确率列表
    epoch_fuzzy_accuracies = []

    # 分析每个epoch
    for epoch_data in epochs_data:
        # 使用正则表达式找到所有的状态和动作对
        actions_data = re.findall(r"\(([0-9]+), ([0-9]+)\),", epoch_data)

        # 初始化计数器
        total_actions = len(actions_data)
        fuzzy_rewarded_sum = 0
        base = 0

        # 检查每个动作的模糊rewarded程度
        for state, action in actions_data:
            state, action = int(state), int(action)
            ideal_action = rewarded_actions.get(state)
            if ideal_action is not None:
                # 计算模糊隶属度
                membership = calculate_reward_membership(action, ideal_action)
                fuzzy_rewarded_sum += membership
                base += 1

        # 计算模糊准确率并添加到列表
        fuzzy_accuracy = fuzzy_rewarded_sum / base if base > 0 else 0
        epoch_fuzzy_accuracies.append(fuzzy_accuracy)

    return epoch_fuzzy_accuracies



def mountaincar_state_distance(current_state, ideal_state):
    distance = abs(current_state[0] - ideal_state[0])
    # 相似度是距离的递减函数
    # 使用指数递减
    # similarity = np.exp(-distance * 10)
    return distance

def mountaincar_state_similarity(current_state, ideal_state):
    distance = abs(current_state[0] - ideal_state[0])
    # 相似度是距离的递减函数
    # 使用指数递减
    similarity = np.exp(-distance * 10)
    return similarity

def mountaincar_action_similarity(current_action, ideal_action):
    similarity = max(1 - abs(current_action - ideal_action), 0)
    return similarity
    return


def parse_mountaincar_log_file(log_file_path):
    rewarded_actions_dict = {}
    epoch_fuzzy_accuracies = []

    try:
        with open(log_file_path, 'r') as file:
            log_content = file.read()

        # 解析rewarded_actions部分
        rewarded_actions_match = re.search(r"rewarded_actions(\{.*?\})", log_content, re.DOTALL)
        if rewarded_actions_match:
            rewarded_actions_str = rewarded_actions_match.group(1)
            rewarded_actions_str = re.sub(r", dtype=float32", "", rewarded_actions_str)
            rewarded_actions_str = re.sub(r"array\((.*?)\)", r"\1", rewarded_actions_str)
            rewarded_actions_dict = ast.literal_eval(rewarded_actions_str)

        # 解析每个epoch的数据
        epochs_data = re.split(r"epoch: \d+\n", log_content)
        if epochs_data:
            epochs_data.pop(0)  # 移除第一个元素，它是rewarded_actions之前的内容

        for epoch_data in epochs_data:
            pattern = r"array\(\[(.*?),\s+(.*?)\],\s+dtype=float32\),\s+array\(\[(.*?)\],\s+dtype=float32\)"
            matches = re.findall(pattern, epoch_data)
            actions_state_data = [(tuple(map(float, match[:2])), [float(match[2])]) for match in matches]

            # print(len(actions_state_data))

            epoch_sim = 0
            base = 0
            for action_state_data in actions_state_data:
                closest_state = min(rewarded_actions_dict.keys(),
                                    key=lambda s: mountaincar_state_distance(action_state_data[0], s))
                if abs(action_state_data[0][0] - closest_state[0]) < 0.1:
                    state_sim = mountaincar_state_similarity(action_state_data[0], closest_state)  # 用指数函数计算相似度
                    action_sim = mountaincar_action_similarity(action_state_data[1][0], rewarded_actions_dict[closest_state][0])
                    # 使用状态相似度和动作相似度来计算奖励
                    fuzzy_sim = state_sim * action_sim
                    epoch_sim += fuzzy_sim
                    base += 1

            if base != 0:
                epoch_fuzzy_accuracies.append(epoch_sim/base)
            else:
                epoch_fuzzy_accuracies.append(0)

    except Exception as e:
        print(f"An error occurred: {e}")

    return epoch_fuzzy_accuracies


if __name__ == '__main__':
    result = parse_mountaincar_log_file('mountaincar/mountaincar_bugfree/time_2024-02-29round_1')
    # print(len(result))
    print(result)
