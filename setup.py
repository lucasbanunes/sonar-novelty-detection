import os
import sys
import platform
import socket

def setup():
    os_name = platform.system()
    host_name = socket.gethostname()

    if os_name == 'Linux':
        if host_name == 'DESKTOP-IORJECJ':
            sys.path.append('/home/lucas_nunes/Repositories/data-analysis')
            lofar_folder = '/home/lucas_nunes/Documentos/datasets/lofar_data'
            output_dir = '/mnt/d/Workspace/output/sonar'
            raw_data = '/mnt/d/datasets/runs_info'
        elif host_name == 'lucas-ideapad-320-15IKB':
            sys.path.append('/home/lucas_nunes/Repositories/data-analysis')
            lofar_folder = '/home/lucas_nunes/Documentos/datasets/lofar_data'
            output_dir = '/home/lucas_nunes/Documentos/sonar_output'
            raw_data = '/home/lucas_nunes/Documentos/datasets/runs_info'
        else:   #Working on Docker
            sys.path.append('/home/data-analysis')
            lofar_folder = '/home/datasets/lofar_data'
            output_dir = '/home/sonar_output'
            raw_data = '/home/datasets/runs_info'

    elif os_name == 'Windows':
        if host_name == 'DESKTOP-IORJECJ':
            sys.path.append('D:\\Repositories\\data-analysis')
            lofar_folder = 'D:\\datasets\\lofar_data'
            output_dir = 'D:\\Workspace\\output\\sonar'
            raw_data = 'D:\\datasets\\runs_info'
        else:
            raise UnknownDesktopError(f"{host_name} desktop doesn't have a setup defined.")
    else:
        raise UnsupportedOS(f'{os_name} is not supported.')

    return lofar_folder, output_dir, raw_data

class UnknownDesktopError(Exception):
    def __init__(self, message):
        super().__init__(message)

class UnsupportedOS(Exception):
    def __init__(self, message):
        super().__init__(message)
