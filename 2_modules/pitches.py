# functions for defining tone intervals and sequences...
import numpy as np

# ==========================================================
# calculate frequecy for intervals (equal temperament!)
# k is the integer element in a chromatic scale
# v is the shift in octave, up(+) or down(-)
# f0 is the root note of the scale. 
def note2freq(k,v,f0):
    freqs = np.round(f0*2**(v+k/12),2)
    return freqs


# ==========================================================
# data base for pitches to connect names to pitches, 
# C4 is just a handy reference note: 
def pitch_dict():
    C4 = 440.0 * 2**(3/12-1)
    # print('C4 = '+str(C4))

    pitch_classes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    ints = np.arange(12)
    v_ref = 4
    v_vec = [-2,-1,0,1,2]

    names_ref = []
    pitches_ref = []
    for dv in v_vec:
        oct = v_ref + dv
        #print('octave = ' + str(oct))
        ref_scale_freqs = note2freq(ints,dv,C4)
        for ind,pname in enumerate(pitch_classes):
            notename = pname + str(oct)
            names_ref.append(notename)
            pitches_ref.append(np.round(ref_scale_freqs[ind],2))
            
    NameFreq_dict = dict((zip(names_ref,np.round(pitches_ref,2))))
    return NameFreq_dict


# ==========================================================
# Define the modes / keys ! 
#>>> d = {}
#>>> d['dict1'] = {}
#>>> d['dict1']['innerkey'] = 'value'
#>>> d
#{'dict1': {'innerkey': 'value'}}

def modes():
    modes_dict = {}
    modes_dict['modes7'] = {
        'ionian':[2,2,1,2,2,2,1],
        'dorian':[2,1,2,2,2,1,2],
        'phrygian':[1,2,2,2,1,2,2],
        'lydian':[2,2,2,1,2,2,1],
        'mixolydian':[2,2,1,2,2,1,2],
        'aeolian':[2,1,2,2,1,2,2],
        'lochrian':[1,2,2,1,2,2,2]
    }

    modes_dict['modes8'] = {
        '21':[2,1,2,1,2,1,2,1],
        '12':[1,2,1,2,1,2,1,2]
    }
    
    #modes_dict = {'modes7':modes7, 'modes8':modes8}
    return modes_dict