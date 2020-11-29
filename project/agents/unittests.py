import unittest
from project.agents.agent import distance_from_ball


class GreedyAgentMethodTests(unittest.TestCase):

    def test1_distance_from_ball(self):
        pos_x = 1
        pos_y = 1
        direction = 0
        ball_x = 3
        ball_y = 3
        dist = distance_from_ball(pos_x, pos_y, direction, ball_x, ball_y)
        self.assertEqual(dist, 5)

    def test2_distance_from_ball(self):
        pos_x = 1
        pos_y = 1
        direction = 2
        ball_x = 3
        ball_y = 3
        dist = distance_from_ball(pos_x, pos_y, direction, ball_x, ball_y)
        self.assertEqual(dist, 6)

    def test3_distance_from_ball(self):
        pos_x = 1
        pos_y = 3
        direction = 0
        ball_x = 3
        ball_y = 3
        dist = distance_from_ball(pos_x, pos_y, direction, ball_x, ball_y)
        self.assertEqual(dist, 2)

    def test4_distance_from_ball(self):
        pos_x = 1
        pos_y = 3
        direction = 2
        ball_x = 3
        ball_y = 3
        dist = distance_from_ball(pos_x, pos_y, direction, ball_x, ball_y)
        self.assertEqual(dist, 4)


if __name__ == '__main__':
    unittest.main()
