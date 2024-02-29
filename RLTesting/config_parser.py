from configparser import ConfigParser
import os

def parserConfig():
    cfg = ConfigParser()
    # 获取当前脚本文件的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建config.ini文件的绝对路径
    config_path = os.path.join(script_dir, 'config.ini')

    # 读取配置文件
    cfg.read(config_path)
    config = {}
    config['root_dir'] = cfg.get('param', 'root_dir')
    # 其他配置项...

    return config

if __name__ == '__main__':
    parserConfig()