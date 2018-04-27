import random
from psychopy import visual, event, core
import math


# Collect responses
def get_typed_answer(win, answer_text):
    answer_text.text = ''
    while True:
        key = event.waitKeys()[0]
        # Add a new number
        if key in '1234567890':
            answer_text.text += key

        # Delete last character, if there are any chars at all
        elif key == 'backspace' and len(answer_text.text) > 0:
            answer_text.text = answer_text.text[:-1]

        # Stop collecting response and return it
        elif key == 'return':
            return(answer_text.text)

        # Show current answer state
        answer_text.draw()
        win.flip()
    return win, answer_text


def run_experiment(win, answer_text, n_dots, contrast):
    dot_xys = []

    for dot in range(n_dots):
        dot_x = random.uniform(-250, 250)
        dot_y = random.uniform(-250, 250)
        dot_xys.append([dot_x, dot_y])


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
    core.wait(4)

    # Collect response
    return (get_typed_answer(win, answer_text))

if __name__=="__main__":
    win = visual.Window(
    size=[500, 500],
    units="pix",
    fullscr=False
    )
    answer_text = visual.TextStim(win)
    guess = run_experiment(win, answer_text, 10, .90)
    print(guess)

