## Coding Standards

This document provides general guidance for developing code for the project.
The rules in this document guide us in writing high-quality code that will
allow us to scale the project and ensure that the code base remains readable and
maintainable.

### Follow the LLVM and Facebook coding standards

Use the Facebook coding standards when writing c++ code. The Facebook coding
standards are almost identical to the LLVM coding standards, except for lower
case variable naming and the underscore suffix. The LLVM coding standards also
covers topics that are compiler specific:

  '''
  http://llvm.org/docs/CodingStandards.html
  '''

The compiler uses the LLVM data structures, described in the links below. The
LLVM data structures and utilities are efficient, well documented and battle
hardened.

  '''
  http://llvm.org/docs/ProgrammersManual.html
  '''

The project uses a reasonable subset of C++11. Just like LLVM, the project does
not use exceptions and RTTI.

### Small incremental changes

The project is developed using small incremental changes. These changes can be
small bug fixes or minor tweaks. Other times, these changes are small steps
along the path to reaching larger stated goals. Long-term development branches
suffer from many problems, including the lack of visibility, difficulty of code
review, lack of testing of the branch and merge difficulty. Commits that go into
the project need to be reviewable. This means that commits need to be relatively
small, well documented and self-contained.

### Add tests

Functional changes to the compiler need to include a testcase. Unit tests and
regression tests are critical to the qualification of the compiler. Every bug
fix needs to include a testcase.

### Format your code

We encourage the use of clang-format to enforce code style and formatting.
Commits that only change the formatting of code should go in separate commits.
This makes reviewing the code and inspecting history easier.

It's recommended to use a pre-commit hook to properly format your code prior
committing any changes. Run the following command from the root of the repo
to enable the hook:
`ln -s ../../utils/format.sh .git/hooks/pre-commit`.

### Commit messages

Here are some guidelines about the format of the commit message:

Separate the commit message into a single-line title and a separate body that
describes the change. Make the title short (80 chars) and readable.  In changes
that are restricted to a specific part of the code, include a [tag] at the start
of the line in square brackets, for example, "[docs] ... ".

If the commit fixes an issue in the bug tracking system, include a link or a
task number. When reverting a change make sure to add a short note that
describes why the patch is being reverted.

### Code review

The project relies on code review to maintain the software quality. Review other
people's changes! Anybody is allowed to review code and comment on patches.

All changes, by all developers, must be reviewed before they are committed to
the repository.
