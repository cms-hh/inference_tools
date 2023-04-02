# coding: utf-8

"""
CMS Di-Higgs inference tools.
"""

import os
import re

import law


# meta infos
__author__ = "The CMS HH Team"
__copyright__ = "Copyright 2020-2023"
__credits__ = [
    "Marcel Rieger",
    "Peter Fackeldey",
    "Alexandra Carvalho Antunes De Oliveira",
    "Torben Lange",
]
__contact__ = "https://gitlab.cern.ch/hh/tools/inference"
__license__ = "BSD-3-Clause"
__status__ = "Development"
__version__ = "0.1.0"

# extended version info
m = re.match(r"^(\d+)\.(\d+)\.(\d+)(-.+)?$", __version__)
version = tuple(map(int, m.groups()[:3])) + (m.group(4),)

# environment flags
dhi_remote_job = str(os.getenv("DHI_REMOTE_JOB", "0")).lower() in ("1", "true", "yes")
dhi_has_gfal = str(os.getenv("DHI_HAS_GFAL", "0")).lower() in ("1", "true", "yes")
dhi_htcondor_flavor = os.getenv("DHI_HTCONDOR_FLAVOR", "cern")

# law contrib packages
law.contrib.load(
    "cms", "git", "gfal", "htcondor", "numpy", "slack", "telegram", "root", "tasks", "wlcg",
)

# initialize wlcg file systems once so that their cache cleanup is triggered if configured
if law.config.has_option("outputs", "wlcg_file_systems"):
    wlcg_file_systems = [
        law.wlcg.WLCGFileSystem(fs.strip())
        for fs in law.config.get_expanded("outputs", "wlcg_file_systems", split_csv=True)
    ]

# import gfal2 once when available to pre-load it before anything ROOT related is imported
if dhi_has_gfal:
    import gfal2  # noqa
