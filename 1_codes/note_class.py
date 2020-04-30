# class for each note
import numpy as np
from matplotlib import cm # colormap ! 
cmap = cm.hot
from matplotlib.patches import Circle
#from scipy.signal import resample



class Note():
    def __init__(self, pc, oct_n, freq, t_start, dur, ampfac, framerate, sampfreq, radian):
        self.pc = pc
        self.oct_n = oct_n
        self.sym_name = pc+str(self.oct_n)
        self.freq = freq
        
        self.t_start = t_start
        self.dur = dur
        self.ampfac = ampfac
        self.framerate = framerate
        self.sampfreq = sampfreq
        self.n_frames = self.dur*self.framerate
        self.time_frames = np.linspace(self.t_start,self.t_start+self.dur,int(self.n_frames))
        
        self.amp_env_mov, self.amp_env_sound = self.amp_env_interps()
        y = self.make_oscillator()
        self.beep = y*self.amp_env_sound*ampfac
        
        # the dot:
        self.current_index = 0
        self.radian = radian
        self.x = np.cos(radian)
        self.y = np.sin(radian)
        self.facecolor = cmap(0.2)
        self.alpha = 0.3
        self.dot_radius_current = []
    
    # both:     
    def amp_env_interps(self):
        #amp_env_y = np.asarray([0,0.5,0.95,0])
        #amp_env_x = np.asarray([0.,0.1,0.3,1.0]) 
        amp_env_y = np.asarray([0,0.7,0])
        amp_env_x = np.asarray([0,0.35,1.0]) 
#         mu = np.log(80) # mean
#         sigma = np.log(2.2) # std
#         #print(sigma),print(mu)
#         x = np.linspace(0,300,100)
#         pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))/ (x * sigma * np.sqrt(2 * np.pi)))
#         pdf[np.isnan(pdf)]=0
#         amp_env_y = (pdf-np.min(pdf))/np.max(pdf-np.min(pdf))
#         amp_env_x = np.linspace(0,1,len(pdf))
        
        # movie
        n_fr = int(self.n_frames)
        amp_env_mov = np.interp(np.linspace(0,1,n_fr),amp_env_x, amp_env_y)
        # sound
        npts = int(self.dur*self.sampfreq)
        amp_env_sound = np.interp(np.linspace(0.0,1.0,npts),amp_env_x, amp_env_y)
        return amp_env_mov, amp_env_sound
        
    # SOUND:
    def make_oscillator(self):
        f = self.freq
        dur = self.dur
        n_cycles = f*dur # [cycles/sec]*[sec]
        fs = self.sampfreq # just make it at 44100.. why not?! 
        npts = int(fs*dur)
        x = np.linspace(0,2*np.pi*n_cycles,npts)
        y = np.sin(x)
        return y
    
    # DOTS
    # make the dot through time
    #def drawdot(self):
    
    def updateDot(self):
        
        if(self.current_index < len(self.amp_env_mov)-1):
            self.current_index += 1

            rfac = 17
            self.dot_radius_current = int(self.amp_env_mov[self.current_index]*rfac)
            # constant values (but can change!)
            x=self.x
            y=self.y
            dot = Circle((x,y),radius=self.dot_radius_current,facecolor=self.facecolor,edgecolor="black",alpha=self.alpha)
            
            return dot