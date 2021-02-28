import numpy as np
from gym_multigrid.rendering import *
from gym_multigrid.multigrid import COLORS, AGENT_COLORS

CELL_SIZE = 32

def render_cell(multigrid, cell_type):

    img = np.zeros(shape=(CELL_SIZE * 3, CELL_SIZE * 3, 3), dtype=np.uint8)

    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

    if cell_type == -3:
        # wall
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS['grey'])
    elif cell_type == -1:
        # ball
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS["green"])
    elif cell_type >= 0:
        # agent
        direction = multigrid.agent_directions[cell_type]
        c = AGENT_COLORS[multigrid.agent_players[cell_type].agent_type]
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * direction)
        fill_coords(img, tri_fn, c)

    img = downsample(img, 3)

    return img


def render_multigrid_as_img(multigrid):

    width_px = multigrid.size * CELL_SIZE
    height_px = multigrid.size * CELL_SIZE

    img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

    for j in range(0, multigrid.size):
        for i in range(0, multigrid.size):

            cell_img = render_cell(multigrid, multigrid.board[i][j])

            ymin = j * CELL_SIZE
            ymax = (j + 1) * CELL_SIZE
            xmin = i * CELL_SIZE
            xmax = (i + 1) * CELL_SIZE
            img[ymin:ymax, xmin:xmax, :] = cell_img

    return img
