import os
from IPython.lib import passwd

c.NotebookApp.ip = '*'
c.NotebookApp.port = int(os.getenv('PORT', 5555))
c.NotebookApp.open_browser = False
c.NotebookApp.notebook_dir = '/mnt/data'
c.MultiKernelManager.default_kernel_name = 'python3'

c.Application.log_level = 'DEBUG'
c.Session.debug = True
