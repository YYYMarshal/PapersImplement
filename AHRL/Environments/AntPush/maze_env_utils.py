class Move(object):
  X = 11
  Y = 12
  Z = 13
  XY = 14
  XZ = 15
  YZ = 16
  XYZ = 17


def can_move_x(movable):
  return movable in [Move.X, Move.XY, Move.XZ, Move.XYZ]


def can_move_y(movable):
  return movable in [Move.Y, Move.XY, Move.YZ, Move.XYZ]


def can_move_z(movable):
  return movable in [Move.Z, Move.XZ, Move.YZ, Move.XYZ]


def can_move(movable):
  return can_move_x(movable) or can_move_y(movable) or can_move_z(movable)


def construct_maze():

  structure = [
        [1, 1,  1,  1,   1],
        [1, 0, 'r', 1,   1],
        [1, 0,  Move.XY, 0,  1],
        [1, 1,  0,  1,   1],
        [1, 1,  1,  1,   1],
    ]

  return structure
