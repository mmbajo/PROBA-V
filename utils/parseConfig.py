from typing import List, Dict
import os


def parseConfig(path: str) -> List[Dict]:
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
            if moduleDefs[-1]['type'] == 'convolutional':
                moduleDefs[-1]['batchNormalize'] = 0
                moduleDefs[-1]['instNormalize'] = 0
        else:
            key, val = line.split('=')
            key = key.lstrip().rstrip()

    # Check all fields if they are supported
    supported = ['type', 'instNormalize', 'batchNormalize', 'filters', 'size', 'stride', 'pad',
                 'activation', 'layers', 'groups', 'from', 'num', 'jitter',
                 'random', 'stride_x', 'stride_y']

    # Check for unsupported fields
    fields = []
    for x in moduleDefs[1:]:
        [fields.append(k) for k in x if k not in f]
    unsupported = [x for x in fields if x not in supported]
    assert not any(unsupported), 'Unsupported fields {} in {}'.format(unsupported, path)

    return moduleDefs
