#!/usr/bin/env python3

import unittest
import sys


DEFAULT_REPORTING = 'stdout'
# other options: 'log' 'userfile'

if __name__ == '__main__':
    pass

    # TODO Find all the tests in project files
    # (this should be file-system independent; i.e. it works regardless of where I clone FREDcast to)
    # TODO Run those tests
    # (probably unittest.TestSuite.runall())
    # TODO Report results according to default values/arguements from bash
    if '-h' in sys.argv:
        print('Userfile: -t <userfile>\nLogfile: -l\nStdout: -s\nDefault: %s' % DEFAULT_REPORTING)
    elif '-t' in sys.argv:
        userfile = sys.argv[sys.argv.index('-t') + 1]
        # TODO Report results to userfile
    elif '-l' in sys.argv:
        pass  # TODO Report results to tests.log
    elif '-s' in sys.argv:
        pass  # TODO Report results to stdout
    else:
        # TODO Report results according to DEFAULT_REPORTING (could use recursion here)
        pass
