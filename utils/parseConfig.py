from typing import List, Dict
import os


def parseConfig(path: str) -> Dict:
    '''
    Parses model cfg file.
    '''

    if not path.endswith('.cfg'):
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):
        path = 'cfg' + os.sep + path

    # Initialize output
    moduleDefs = []

    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]  # dont read comm
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of whitespaces

    for line in lines:
        if line.startswith('['):
            moduleDefs.append({})
            moduleDefs[-1]['type'] = line[1:-1].lstrip().rstrip()
        else:
            key, val = line.split('=')
            key = key.lstrip().rstrip()

            if moduleDefs[-1]['type'] == 'Preprocessing':
                if 'ckpt' in key:
                    moduleDefs[-1][key] = [int(x) for x in val.split(',')]
                elif 'low_res_threshold' in key or 'high_res_threshold' in key:
                    moduleDefs[-1][key] = float(val.strip())
                elif 'to_flip' in key or 'to_rotate' in key:
                    moduleDefs[-1][key] = bool(int(val.strip()))
                else:
                    moduleDefs[-1][key] = int(val.strip())

            elif moduleDefs[-1]['type'] == 'Net':
                if 'decay_rate' in key:
                    moduleDefs[-1][key] = float(val.strip())
                elif 'is_grayscale' in key:
                    moduleDefs[-1][key] = bool(int(val.strip()))
                else:
                    moduleDefs[-1][key] = int(val.strip())

            elif moduleDefs[-1]['type'] == 'Train':
                if 'learning_rate' in key or 'split' in key:
                    moduleDefs[-1][key] = float(val.strip())
                elif 'optimizer' in key or 'loss' in key:
                    moduleDefs[-1][key] = val.strip()
                else:
                    moduleDefs[-1][key] = int(val.strip())
            else:
                moduleDefs[-1][key] = val.strip()

    # Check all fields if they are supported
    supported = ['type', 'raw_data', 'preprocessing_out',
                 'model_out', 'batch_size', 'epochs', 'learning_rate', 'optimizer', 'split', 'num_res_blocks',
                 'num_low_res_imgs', 'scale', 'num_filters', 'kernel_size', 'exp_rate', 'decay_rate', 'is_grayscale',
                 'max_shift', 'patch_size', 'patch_stride', 'low_res_threshold', 'high_res_threshold',
                 'num_low_res_permute', 'to_flip', 'to_rotate', 'ckpt', 'test_out', 'train_out', 'loss']

    # Check for unsupported fields
    fields = []
    for x in moduleDefs[1:]:
        [fields.append(k) for k in x if k not in fields]
    unsupported = [x for x in fields if x not in supported]
    assert not any(unsupported), 'Unsupported fields {} in {}'.format(unsupported, path)

    # Convert to one big inputDictionary
    config = {}
    for c in moduleDefs:
        config.update(c)
    del config['type']

    return config
