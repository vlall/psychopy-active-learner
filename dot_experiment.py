import random
from psychopy import visual, event, core
import time
"""
This file runs the dot experiment using the Psychopy Toolbox, which presents
a uniform distribution of dots across a 500x500 screen. It takes as input the
`n_dots` being displayed and the `contrast` of those dots. This is used by the Bayesian
active model selection code in `psychopy_learner.py` which manipulates a combination of these variables
in order to converge to the kernel grammar which best describes the relationship between stimuli and
human behavior.
"""
def get_typed_answer(win, guess):
    instruction_text = visual.TextStim(win, text = u'How many dots did you see?', pos=(0, 100))
    guess.text = ''
    timeout = time.time() + 5
    while True:
        key = event.waitKeys()[0]
        # Add a new number
        if key in '1234567890':
            guess.text += key

        # Delete last character, if there are any chars at all
        elif key == 'backspace' and len(guess.text) > 0:
            guess.text = guess.text[:-1]

        # Stop collecting response and return it
        elif key == 'return':
            return (guess.text)

        # Show current answer state
        instruction_text.draw()
        guess.draw()
        win.flip()

def run_experiment(win, guess, n_dots, contrast):
    dot_xys = []

    for dot in range(n_dots):
        dot_x = random.uniform(-250, 250)
        dot_y = random.uniform(-250, 250)
        dot_xys.append([dot_x, dot_y])
    #print(dot_xys)

    dot_stim = visual.ElementArrayStim(
        win=win,
        units="pix",
        elementTex=None,
        elementMask="circle",
        sizes=10,
        contrs=contrast,
        nElements = n_dots,
        xys = dot_xys,
    )

    dot_stim.draw()
    win.flip()

    # Collect response
    return (get_typed_answer(win, guess))


if __name__=="__main__":
    """
    Run a test of the experiment.
    """
    win = visual.Window(
    size=[500, 500],
    units="pix",
    fullscr=False
    )
    answer_text = visual.TextStim(win)
    guess = run_experiment(win, answer_text, 10, .90)
    print(guess)

