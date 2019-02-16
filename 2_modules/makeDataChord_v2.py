# make chords from the data.. output of owecianizer...
# a better approach 
# pitches are the list of notes to find in the data(mapped to frequency)
# notes is the list of when they occur in the data
import numpy as np

def makeDataChord(pitches,time,times,data_notes):
    t_end = time[-1]
    ch_notes = []
    ch_times = []
    ch_durs = []

    for i,note in enumerate(pitches):
        note_times = []
        note_times = times[data_notes==note]
        print(note_times)
        print('now looping over the next note_times:')
        
        for ind_nt, t in enumerate(note_times):
            # find the index at this point.
            print('time = ' + str(t))
            # itemindex = numpy.where(array==item)
            ind_time = 0
            ind_time = np.where(times==t)
            ind_time = int(ind_time[0])
            print('ind_time = ' + str(ind_time))
            dn_next = data_notes[ind_time+1]
            
            # positive slope
            if dn_next > data_notes[ind_nt]:
                if ind_nt == (len(note_times)-1):
                    dur = t_end-t             
                else:
                    dur = note_times[ind_time+1] - t  #note_times[ind_time]
                    
                    ch_notes.append(note)
                    ch_times.append(t)
                    ch_durs.append(dur)
                    
            # negative slope      
            if dn_next < data_notes[ind_time]:
                if ind_nt==0:
                    dur = t
                    ch_notes.append(note)
                    ch_times.append(time[0])
                    ch_durs.append(dur)
                    
        
    return ch_notes, ch_times, ch_durs
        
#         if len(a)%2==0:
#             print('even')
#             for i in range(len(a)):
#                 if (i)%2==0:
#                     dur_a = a[i+1]-a[i]
#                     ch_notes.append(note)
#                     ch_times.append(a[i])
#                     ch_durs.append(dur_a)
                    
#         elif len(a)%2==1:
#             print('odd')
#             for i in range(len(a)):
#                 if i<len(a)-1:  #  no -1, it breaks; -1 gives same as -2 
#                     if (i)%2==0: 
#                         #dur_a = a[i+1]-a[i]
#                         dur_a = a[i+1]-a[i]
#                         ch_notes.append(note)
#                         ch_times.append(a[i])
#                         ch_durs.append(dur_a)
#                 if len(a)==1:
#                     #dur_a = time[-1] - times[i]
#                     dur_a = time[-1] - times[i]
#                     ch_notes.append(note)
#                     ch_times.append(a[i])
#                     ch_durs.append(dur_a)
