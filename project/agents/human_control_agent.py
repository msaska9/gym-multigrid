from project.agents.agent import Agent
import tkinter as tk
from PIL import Image, ImageTk
import time

root = tk.Tk()
root.title("Human Agent")
root.configure(background='black')
root.geometry("400x80")
root.withdraw()


def create_img(path):
    img = Image.open(path)
    img = img.resize((50, 50), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    return img


still_image = create_img("../images/still.jpg")
left_image = create_img("../images/left.jpg")
right_image = create_img("../images/right.jpg")
forward_image = create_img("../images/forward.jpg")
pickup_image = create_img("../images/pickup.jpg")


def set_reply(answer, value, tk_var):
    answer[0] = value
    tk_var.set(1)


def ask_for_human_move():
    # root.deiconify()
    tk_var = tk.IntVar()
    answer = [0]

    button = tk.Button(root, text="", image=still_image, command=lambda: set_reply(answer, 0, tk_var))
    button.place(relx=.1, rely=.5, anchor="c")

    button = tk.Button(root, text="", image=left_image, command=lambda: set_reply(answer, 1, tk_var))
    button.place(relx=.3, rely=.5, anchor="c")

    button = tk.Button(root, text="", image=right_image, command=lambda: set_reply(answer, 2, tk_var))
    button.place(relx=.5, rely=.5, anchor="c")

    button = tk.Button(root, text="", image=forward_image, command=lambda: set_reply(answer, 3, tk_var))
    button.place(relx=.7, rely=.5, anchor="c")

    button = tk.Button(root, text="", image=pickup_image, command=lambda: set_reply(answer, 4, tk_var))
    button.place(relx=.9, rely=.5, anchor="c")

    button.wait_variable(tk_var)

    # root.withdraw()

    return answer[0]


class HumanAgent(Agent):
    def __init__(self, agent_id, colour=4):
        super().__init__(agent_id, agent_type=colour)
        root.deiconify()

    def start_simulation(self, observation, rounds):
        """ Nothing to be done """

    def next_action(self, observation, reward, round_id):
        return ask_for_human_move()

    def end_simulation(self, observation, reward, round_id, learn_from=True):
        """ Nothing to be done """


if __name__ == '__main__':
    print(ask_for_human_move())
    print(ask_for_human_move())
