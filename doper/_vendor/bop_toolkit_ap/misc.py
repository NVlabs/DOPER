# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague
# Modified from upstream; see NOTICE.

"""Miscellaneous functions."""

import sys
import datetime


def log(s):
    """A logging function.

    :param s: String to print (with the current UTC date and time).
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    utc_now_str = f"{now.month}/{now.day}|{now.hour:02d}:{now.minute:02d}:{now.second:02d}"
    sys.stdout.write(f"{utc_now_str}: {s}\n")
    sys.stdout.flush()
