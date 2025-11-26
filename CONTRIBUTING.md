# Contributing to RFDpoly
Thank you for helping RFDpoly grow! Please follow the
guidelines below to ensure the consistency of the code
and documentation quality.

To contribute to RFDpoly - either through code or 
documentation contributions - please fork the repository,
make changes, and create a pull request (PR). The pull 
request should have `main` as the base branch.

If your PR is a result of an Issue or Discussion, make
sure to mention it in your description. 

## Issues vs. Discussions
An issue should be opened if there is an issue with the 
code or there is a request for a new feature. 

The Discussions section should be used to answer questions
about best practices, usage, etc. 

## Code Contributions

### Code Style and Quality
When contributing to the code base, please adhere to the [PEP8 style guide](https://peps.python.org/pep-0008/).

### Tests
Please write tests for any contributed code or append to 
existing tests.

Existing tests can be found in `rf_diffusion/test_inference_commands.py`

### Approving and Merging Pull Requests
Currently only the original developers and the Devel Team 
at Rosetta Commons can approve and merge pull requests. PRs
will be merged via a merge commit, so all commits from the 
feature branch will be added to the base branch.

## Documentation Contributions

To preview the changes you made to the documentation before making a Pull Request you will need to
install a few dependencies which can be done via
```
uv pip install -r docs/docs_requirements.txt
```

To build the documentation, navigate to `docs`  and run: 
```
make html
```
You will then be able to see the generated HTML pages in `build/html`
