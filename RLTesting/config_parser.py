from configparser import ConfigParser

def parserConfig():
    cfg = ConfigParser()
    cfg.read('config.ini')
    config = {}
    config['root_dir'] = cfg.get('param','root_dir')
    # config['real_life'] = cfg.get('param','real_life')

    # config['bug_num'] = int(cfg.get('param','bug_num'))
    # config['specific_bug'] = cfg.get('param','specific_bug')
    
    temp = cfg.get('param','specified_bug_id')[1:-1]
    if temp == '':
        config['specified_bug_id'] = []
    else: 
        config['specified_bug_id'] = [int(t.strip()) for t in temp.split(',')]

    config['epoches'] = int(cfg.get('param','epoches'))
    config['rounds'] = int(cfg.get('param','rounds'))
    config['model_type'] = cfg.get('param','model_type')
    return config