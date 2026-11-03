"""Explicit direction interface, developed after v2 center/turn deadlock.

This is a calibration change, not evidence of a new research contribution.
"""
def prompt(previous):
    return ('Find the blue box in the CURRENT image. The previous movement '
            f'({previous}) may be outdated. CENTER means the center of the blue '
            'box lies in the middle 20 percent of the FULL image width. '
            'Otherwise reply LEFT or RIGHT of that central band. '
            'If no blue box is visible reply ABSENT. '
            'Reply with exactly one word: LEFT, CENTER, RIGHT, or ABSENT.')

# Exactly the prompt frozen in point-contract-calibration/protocol.json.
# This fresh visual direction replacement does not consume prior guidance.
POINT_PROMPT=('Locate the center of the blue box in the image. Return its image coordinates '
        'normalized from 0 to 1000, where x=0 is the left edge, x=1000 the right edge, '
        'y=0 the top and y=1000 the bottom. Reply only with JSON: {"point":[x,y]}. '
        'If no blue box is visible, reply {"point":null}.')
