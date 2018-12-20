import sys
import os
import copy

## Class that represents a single CHECK statement.
class Check:
    ## @param str is the text from one CHECK statement line.
    def __init__(self, str):
        self.is_beginning = False
        self.is_ending = False
        self.str = str

        if str.startswith("{{^}}"):
            self.is_beginning = True
            self.str = self.str[len("{{^}}"):]

        if str.endswith("{{$}}"):
            self.is_ending = True
            self.str = self.str[:-len("{{$}}")]

    ## Look for check pattern in a line starting at @param start_idx.
    # start_idx defaults to 0 if it's not specified. @returns the beginning
    # index of the pattern if found otherwise returns -1.
    def find(self, line, start_idx):
        if start_idx == None:
            start_idx = 0

        strIdx = line.find(self.str, start_idx)

        if strIdx == -1:
            return -1

        if self.is_beginning:
            # Not at beginning if str not found at first index of line.
            if strIdx != 0:
                return -1

        if self.is_ending:
            afterStrIdx = strIdx + len(self.str)
            # Not at end if more characters are remaining and the the next
            # characters are not linesep.
            if (len(line) != afterStrIdx and
                line.find(os.linesep, afterStrIdx) != afterStrIdx
               ):
                return -1

        return strIdx

# Reads in a file at @file_path and @returns a list of Checks.
def parse_checks(file_path):
    checks = []
    file = open(file_path, "r")
    for line in file:
        if line.startswith("CHECK"):
            checks.append(Check(line[len("CHECK:"):].strip()))
    return checks


## Match as many checks as possible to the text in @param line. @returns any
# Checks that could not be matched.
def match_line_to_checks(line, checks):
    checkIdx = 0
    lineIdx = 0
    for check in checks:
        idx = check.find(line, lineIdx)
        if idx == -1:
            break
        checkIdx += 1
        lineIdx = idx
    return checks[checkIdx:]

## Read stdin and match against Checks, @returns a list of Checks that were not
# able to be matched.
def run_checks(checks):
    checks = copy.deepcopy(checks)
    for line in sys.stdin:
        checks = match_line_to_checks(line, checks)
        if len(checks) == 0:
            break
    return checks

## Reads a file specified by command line argument and verifies that each CHECK
# statement in that file is seen in stdin in order.
def main():
    # Get path from command line to file to parse CHECK statements from.
    if len(sys.argv) != 2:
        print "Received {}/{} expected command line arguments. Requires path " \
               "to file to compare against.".format(len(sys.argv) -1 , 1)
        sys.exit(1)
    file_path = os.path.abspath(sys.argv[1])

    # Parse checks.
    checks = parse_checks(file_path)

    # Run all checks against stdin.
    remaining_checks = run_checks(checks)

    # If any checks couldn't be matched then exit with failure.
    if len(remaining_checks) > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
