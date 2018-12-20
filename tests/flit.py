import os
import copy
import sys
import tempfile
import subprocess
import imp
from sets import Set

## Flit config class populated by cfg.py files.
class Config:
    def __init__(self):
        self.substitutions = []
        self.available_features = Set()
        self.suffixes = []

## Represents one test, contains the test name and commands to run as well as
# information about requirements that weren't which will prevent this test from
# running.
class Test:
    def __init__(self, test_name):
        self.name = test_name
        self.run_commands = []
        self.unmet_requirements = []

    # Execute run commands.
    def run(self):
        assert len(self.unmet_requirements) == 0, \
            "Shouldn't run test with unmet requirments."
        devnull = open(os.devnull, 'w')
        for run_command in self.run_commands:
            retval = subprocess.call(run_command, shell=True, stdout=devnull)
            if retval != 0:
                return (retval, run_command)

        # If no problems ocurred then return error code 0.
        return (0, "")

## Call str.replace(before, after) with all pairs of (before, after) in
# replacements.
def string_replace_all(str, replacements):
    newStr = str
    for before, after in replacements:
        newStr = newStr.replace(before, after)
    return newStr

## Given @param path_to_test, open and parse that test file into a Test object.
def parse_test(config, path_to_test, test_name):
    test = Test(test_name)
    file = open(path_to_test, "r")
    for line in file:
        if line.startswith("RUN"):
            text = line[len("RUN:"):].strip()
            test.run_commands \
                .append(string_replace_all(text, config.substitutions))
        if line.startswith("REQUIRES"):
            text = line[len("REQUIRES:"):].strip()
            text = string_replace_all(text, config.substitutions)
            if not text in config.available_features:
                test.unmet_requirements.append(text)

    return test

## Create a Config instance for a specific test. Each test gets its own config
# because of the specific substitutions it has.
def make_test_config(config, test_dir_name, test_full_path):
    test_config = copy.deepcopy(config)
    test_config.substitutions.append(("%s", test_full_path))
    test_config.substitutions.append(("%S", test_dir_name))
    test_config.substitutions.append(("%p", test_dir_name))
    test_config.substitutions.append(("%{pathsep}", os.path.sep))
    temp_file_name = test_full_path.replace(os.path.sep, "_")
    test_config.substitutions \
        .append(("%t", os.path.join(config.temp_dir, temp_file_name)))
    return test_config

## Traverses directory tree rooted at config.test_source_root collecting tests
# to run.
# TODO support updates to configs specified by lit.local.cfg files.
def collect_tests(config):
    tests = []
    for dir_name, subdirList, fileList in os.walk(config.test_source_root):
        for fname in fileList:
            for suffix in config.suffixes:
                if fname.endswith(suffix):
                    test_full_path = \
                        os.path.abspath(os.path.join(dir_name, fname))
                    test_name = fname[:-len(suffix)]
                    test_config = \
                        make_test_config(config, dir_name, test_full_path)
                    tests.append(parse_test(test_config, test_full_path, test_name))
    return tests

## Run all tests.
def run_tests(config, tests):
    curDir = os.getcwd()
    if config.test_exec_root != None:
        os.chdir(config.test_exec_root)
    allPassed = True
    i = 0
    numTests = len(tests)
    for test in tests:
        i += 1

        # Check requirements and skip test if not all requirements met.
        if len(test.unmet_requirements) != 0:
            print "SKIP: {} ({} of {})".format(test.name, i, numTests)
            print "******************** TEST {} SKIPPED ********************" \
                .format(test.name)
            print "Test \"{}\" skipped due to unmet requirement(s): \"{}\"" \
                .format(test.name, ", ".join(test.unmet_requirements))
            print "********************"
            continue

        # Run test.
        (retval, faled_run_command) = test.run()

        # Print test run results.
        if retval == 0:
            print "PASS: {} ({} of {})".format(test.name, i, numTests)
        else:
            allPassed = False
            print "FAIL: {} ({} of {})".format(test.name, i, numTests)
            print "******************** TEST {} FAILED ********************" \
                .format(test.name)
            print "Test \"{}\" exited with code {} while running command \"{}\"" \
                .format(test.name, retval, faled_run_command)
            print "********************"
            continue

    os.chdir(curDir)
    return allPassed

## Read in each config file specified in the arguments to this script.
def build_config():
    config = Config()
    numReceivedCLArgs = len(sys.argv)
    for i in range(1, numReceivedCLArgs):
        config_path = os.path.abspath(sys.argv[i])
        config_source = imp.load_source('config_source', config_path)
        import config_source
        config_source.setup_config(config)
    return config

## Builds a Config from config files specfied in command line args then runs
# all tests specified by the config.
def main():
    config = build_config()

    assert config.test_source_root != None, "A test source root must be set."

    # Create a temp dir.
    temp_dir = tempfile.mkdtemp()

    config.temp_dir = temp_dir

    # Find and run all tests.
    tests = collect_tests(config)

    all_tests_passed = run_tests(config, tests)

    # Clean up the temp dir.
    subprocess.call(["rm", "-rf", temp_dir])

    if not all_tests_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
