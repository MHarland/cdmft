import matplotlib
matplotlib.use("PDF")
from matplotlib import pyplot as plt


"""
this file may change matplotlib.rc
the following package changes matplotlib.rc, but is not public available, since it changes very frequently
"""
try:
    from mpl_to_latex.matplotlib_to_latex import set_log_parameters as set_mpl
    set_mpl()
except:
    pass
fig = plt.figure()
ax = fig.add_axes([.13,.13,.84,.82])
