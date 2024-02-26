import re
import ast
import os


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


def calculate_reward_membership(action, ideal_action, fuzziness=0.5):
    """
    计算一个动作相对于理想动作的模糊隶属度。
    Args:
        action: 实际采取的动作
        ideal_action: 理想的动作（被认为是rewarded的动作）
        fuzziness: 模糊度参数，定义了非理想动作的惩罚程度

    Returns:
        float: 表示隶属度的数值
    """
    distance = abs(action - ideal_action)
    if distance != 0:
        distance = 0.99
    return max(1 - fuzziness * distance, 0)

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

        # 检查每个动作的模糊rewarded程度
        for state, action in actions_data:
            state, action = int(state), int(action)
            ideal_action = rewarded_actions.get(state)
            if ideal_action is not None:
                # 计算模糊隶属度
                membership = calculate_reward_membership(action, ideal_action)
                fuzzy_rewarded_sum += membership

        # 计算模糊准确率并添加到列表
        fuzzy_accuracy = fuzzy_rewarded_sum / total_actions if total_actions > 0 else 0
        epoch_fuzzy_accuracies.append(fuzzy_accuracy)

    return epoch_fuzzy_accuracies