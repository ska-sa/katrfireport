import os
from pkg_resources import resource_filename

def get_parameter_file(default_file):
    parameter_dir = resource_filename(__name__, 'conf')
    filename = dafault_file
    parm_file = os.path.join(parameter_dir, filename)
    return parm_file
