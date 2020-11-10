from colorama import Fore, Back, Style
import sys

PURPLE = '\033[95m'
RED   = "\033[1;31m"
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
YELLOW = '\033[93m'
BOLD    = "\033[;1m"
UNDERLINE = '\033[4m'
ENDC = '\033[0;0m'

def purple(s):
    return PURPLE + s + ENDC

def red(s):
    return RED + s + ENDC

def blue(s):
    return BLUE + s + ENDC

def cyan(s):
    return CYAN + s + ENDC

def green(s):
    return GREEN + s + ENDC

def yellow(s):
    return YELLOW + s + ENDC

def bold(s):
    return BOLD + s + ENDC

def underline(s):
    return UNDERLINE + s + ENDC

def DEBUG(s):
    return cyan("[DEBUG] "+s)

def INFO(s):
    return yellow("[INFO] "+s)

def WARNING(s):
    return yellow("[WARN] "+s)

def ERROR(s):
    return red("[ERROR] "+s)

def DONE(s):
    return green("[DONE] "+s)

def inline(s, fmt=None):
    if fmt is not None:
        s = fmt(s)
    sys.stdout.write(s)
    sys.stdout.flush()
