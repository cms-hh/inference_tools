# HH Inference Tools

[![Documentation badge](https://img.shields.io/badge/Documentation-passing-brightgreen)](http://cern.ch/cms-hh/tools/inference) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Introduction

Connect to a CentOS 7 node on lxplus:
```shell
ssh <cern_username>@lxplus.cern.ch
```

Clone this repository:
```shell
git clone https://gitlab.cern.ch/hh/tools/inference.git
cd inference
```

Run the following two steps everytime you newly login to a CentOS 7 node on lxplus in order to setup the correct software.

1. Be patient this may take some time if you set up the software for the first time:
```shell
source setup.sh
```

2. Let law index your tasks and their parameters (for autocompletion)
```shell
law index --verbose
```



## Documentation

The documentation is hosted at [cern.ch/cms-hh/tools/inference](https://cern.ch/cms-hh/tools/inference).

It is built with [MkDocs](https://www.mkdocs.org) using the [material](https://squidfunk.github.io/mkdocs-material) theme and support for [PyMdown](https://facelessuser.github.io/pymdown-extensions) extensions.
Developing and building the documentation locally requires docker and a valid login at the CERN GitLab container registry.

To login, run

```shell
docker login gitlab-registry.cern.ch
```

and type your CERN username and password.
Then, to build the documentation, run

```shell
./docs/docker/run.sh build
```

which creates a directory `docs/site/` containing static HTML pages.
To start a server to browse the pages, run

```shell
./docs/docker/run.sh build
```

and open your webbrowser at [http://localhost:8000](http://localhost:8000).
By default, all pages are *automatically rebuilt and reloaded* when a source file is updated.


## For developers

Code style is enforced with the formatter "black": https://github.com/psf/black.
The default line width is increased to 100 (see `pyproject.toml`).

To run the linting (i.e. show locations in the code that require formatting), run

```shell
dha_lint
```

and to automatically fix the formatting, add `fix` to the command.


## Contributors

* Peter Fackeldey: peter.fackeldey@cern.ch (email)
* Marcel Rieger: marcel.rieger@cern.ch (email)
