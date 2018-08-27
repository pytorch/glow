# Title

The title of the PR should be short and expressive enough to reflect the proposed changes.
It should start with the name of the subsystem affected by the PR in the square braces.
For example:
* [quantization] Properly quantize ReLU
* [opencl] Define a type cl_host_size_t exactly matching the type size_t used on the host side
* [Type.h] Add comments for the ElemKinds

Ideally, the same format should be used in the commit message.

# Description

The description field should include a more verbose explanation of what this PR does.
If this PR causes a change in behavior it should document the behavior before and after.
If it fixes a bug, please, describe what the original issue is and how the change resolves it.

# Formatting

Please, run clang-format before submitting the PR.

# Testing

The testing section should include an explanation of what testing was done.
For example unit test, manual testing through stand-alone testing binaries like mnist, cifar10, etc.
Please use your best judgment or ask for guidance if you are unsure what kind of testing is required for a given change.
The riskier the change, the more comprehensive the testing should be.
If PR fixes an existing bug, make sure to present unit test which reveals the original problem.

# Documentation

Please, make sure to update existing docs if your change affects those.
Docs can be found under the glow/docs. Use your best judgment for creating new docs.

# Issues

If the PR fixes an existing issue, please add a line of the form:
* Fixes #Issue.

This way Github will make sure that underlying referenced issue will be automatically closed once PR is merged.

If the purpose of the PR to address issue partially, please, just tag the issue:
* #Issue.

