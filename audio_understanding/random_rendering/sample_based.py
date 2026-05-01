# sample_based.py
import numpy as np
SAMPLE_DUR = 4
RELEASE_DUR = 1
# ['Bass', 'Brass', 'Chromatic Percussion', 'Guitar', 'Organ', 'Piano', 
# 'Pipe', 'Reed', 'Strings', 'Strings (continued)', 'Synth Lead', 'Synth Pad']  

# Index	ID
# 0	bass
# 1	brass
# 2	flute
# 3	guitar
# 4	keyboard
# 5	mallet
# 6	organ
# 7	reed
# 8	string
# 9	synth_lead
# 10	vocal
#* 解决方案是给Slakh的instru_class再压缩

class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
INSTRU_CLASS_MAPPING = {
    'Bass': 0,
    'Brass': 1,
    'Chromatic Percussion': 5,
    'Guitar': 3,
    'Organ': 6,
    'Piano': 4,
    'Pipe': 2,
    'Reed': 7,
    'Strings': 8,
    'Strings (continued)': 8,
    'Synth Lead': 9,
    'Synth Pad': 9,
    'Vocal': 10,
}
def prune_audio(audio, dur, sr):
    ''' first 3 seconds A+D+S, last 1 second R'''
    assert abs(len(audio) // sr - SAMPLE_DUR) <= 0.1
    return_audio = np.zeros(int(SAMPLE_DUR * sr))
    if dur > SAMPLE_DUR:
        return_audio[:len(audio)] = audio 
    else:
        # the release time is the first ratio seconds of the release
        # the attack and sustain time is the first ratio seconds of the attack and sustain
        ratio = dur / SAMPLE_DUR
        RELEASE_TIME = int((SAMPLE_DUR-RELEASE_DUR) * sr * ratio)
        return_audio[-RELEASE_TIME:] = audio[int((SAMPLE_DUR-RELEASE_DUR) * sr):int((SAMPLE_DUR-RELEASE_DUR) * sr) +RELEASE_TIME]
        return_audio[:-RELEASE_TIME] = audio[:int(dur * sr - RELEASE_TIME)]
    return return_audio

def scale_velocity(audio, velocity, target_velocity):
    pass        
def get_note(family_name, pitch, velocity, dur):