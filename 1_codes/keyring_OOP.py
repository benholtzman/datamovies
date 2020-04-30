# make a pitch ring and animate the tones moving around it... 
# and the sound, combine with movie... 
# to illustrate relatively simple OOP

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from subprocess import Popen
import subprocess as sp
import os

import sys
sys.path.append('../2_modules/')
import pitches as pitch_fcns
#import IPython.display as ipd

# ===========================================================
frames_dir = './frames/'
anim_out_name = frames_dir + 'keyring_OOP_test.mp4' 

# =============================================
def makeKeyRing_base(indexes):
    circle = np.linspace(0,2*np.pi,64)
    r = 1.0
    x = r*np.sin(circle)
    y = r*np.cos(circle)

    # the note locations. 
    base_dots = np.linspace(0,2*np.pi,13)
    xd = r*np.sin(base_dots)
    yd = r*np.cos(base_dots)

    # the text locations
    r = 1.15
    xt = r*np.sin(base_dots)
    yt = r*np.cos(base_dots)

    # =================
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')

    # (0) plot a filled square with a filled circle in it...
    # patches.Rectangle((x,y,lower left corner),width,height)
    #ax1.add_patch(patches.Rectangle((0.1, 0.1),0.5,0.5,facecolor="red"))

    ax1.add_patch(patches.Rectangle((-1.25, -1.25),2.5,2.5,facecolor=[0.6, 0.6, 0.6]))
    ax1.plot(x,y,'k-')
    ax1.plot(xd,yd,'w.')

    radius_norm = 0.08  # radius normalized, scaled to size of box

    for ind,interval in enumerate(indexes):
        # print(ind,interval)
        ax1.add_patch(patches.Circle((xd[interval], yd[interval]),radius_norm,facecolor="red")) 
        ax1.text(xt[interval], yt[interval],pitch_classes[interval])

    # add current note !  
    # posind = indexes[indnow]
    # print(posind)
    # ax1.add_patch(patches.Circle((xd[posind], yd[posind]),radius_norm*2,facecolor="yellow",alpha=0.3))

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    return fig1


pitch_classes = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
pc_indexes = [0,2,4,5,7,9,11]
fig1 = makeKeyRing_base(pc_indexes) ;
plt.show()



sys.exit()


# ===============================================

pitch_classes = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
pc_indexes = [0,2,4,5,7,9,11]
# ind=1
# fig1 = makePitchRing(indexes,ind)

# now add the time, assuming 30 frames per second... 
fps = 29.97
durs = np.asarray([2.,2.,2.,2.,2.,2.,2.])
n_frames = durs*fps
print(n_frames)

fr_count = 0
for ind,n_frame in enumerate(n_frames):
    print(ind,pc_indexes[ind])
    for i in range(int(n_frame)):
        fig1 = makePitchRing(pc_indexes,ind) ;
        figname = frames_dir + '{:0>4d}'.format(fr_count) +'.png' #{:0>5d}".format(i)
        fig1.savefig(figname);
        plt.close()
        del fig1
        
        fr_count += 1

# ====================================================
# ASSEMBLE MOVIE from the frames. 
#
assemble_ffmpeg_cmd = 'ffmpeg -r:v '+str(fps)+' -i "' + frames_dir + '%4d.png" -codec:v libx264 "'+ anim_out_name + '"'
print(assemble_ffmpeg_cmd)
make_movie = sp.Popen(assemble_ffmpeg_cmd, shell=True) 
make_movie.wait()

# delete the image files. 
cleanup_cmd = 'rm -r '+frames_dir+'*.png'
cleanup = sp.Popen(cleanup_cmd, shell=True) 
cleanup.wait()