# make chords from the data.. output of owecianizer... 
def makeDataChord(mNotes,time,times,notes):
    ch_notes = []
    ch_times = []
    ch_durs = []

    for i,note in enumerate(mNotes):
        a = []
        a = times[notes==note]
        print(a)
        if len(a)%2==0:
            print('even')
            for i in range(len(a)):
                if (i)%2==0:
                    dur_a = a[i+1]-a[i]
                    ch_notes.append(note)
                    ch_times.append(a[i])
                    ch_durs.append(dur_a)
        elif len(a)%2==1:
            print('odd')
            for i in range(len(a)):
                if i<len(a)-1:
                    if (i)%2==0:
                        #dur_a = a[i+1]-a[i]
                        dur_a = a[i+1]-a[i]
                        ch_notes.append(note)
                        ch_times.append(a[i])
                        ch_durs.append(dur_a)
                if len(a)==1:
                    #dur_a = time[-1] - times[i]
                    dur_a = time[-1] - times[i]
                    ch_notes.append(note)
                    ch_times.append(a[i])
                    ch_durs.append(dur_a)

    return ch_notes, ch_times, ch_durs
