import time, sys
#from IPython.display import clear_output
from IPython.display import display

def updateProgress(progress, steps, dh, prog_ant, tempo_ini):
    if(progress == 0):
        tempo_ini = time.time()
        text = 'Processing...'
        dh = display(text, display_id = True)
        prog_ant = progress
    elif((progress - prog_ant)*100 >= steps or progress == 1):
        tempo = time.time()
        tempo_restante = ((1 - progress)/progress) * (tempo - tempo_ini)
        tempo_restante = round(tempo_restante/60, 2)
        prog_ant = progress
        bar_length = 20
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
        if progress < 0:
            progress = 0
        if progress >= 1:
            progress = 1
        block = int(round(bar_length * progress))
        text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
        text = text + ' Time Left: ' + str(tempo_restante) + ' min' 
        dh.update(text)
    return dh, prog_ant, tempo_ini