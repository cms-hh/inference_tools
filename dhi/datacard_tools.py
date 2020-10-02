# coding: utf-8

"""
Helpers to manipulate and work with datacards and shape files.
"""

import os
import shutil
import contextlib

import law


def read_datacard_blocks(datacard, is_separator=None):
    """
    Reads the content of *datacard*, divides the lines into blocks according to a certain separator
    and returns a list containing lines per block as lists. *is_separator* can be a function
    accepting a line string as its sole argument to check whether a line is a separator. By default,
    a line is treated as a separator when it starts with at least three "-".
    """
    if not callable(is_separator):
        is_separator = lambda line: line.startswith("---")

    blocks = []

    # go through datacard line by line
    datacard = os.path.abspath(os.path.expandvars(os.path.expanduser(datacard)))
    with open(datacard, "r") as f:
        blocks.append([])
        for line in f.readlines():
            line = line.strip()

            # check if we reach a new block
            if is_separator(line):
                blocks.append([])
                continue

            # store the line in the last block
            blocks[-1].append(line)

    return blocks


@contextlib.contextmanager
def manipulate_shape_lines(datacard, target_datacard=None):
    """
    Context manager that opens a *datacard* and yields a list of lines (strings) with the contained
    shape descriptions. The list can be updated in-place to make changes to the datacard. When a
    *target_datacard* is defined, the changes are saved in a new datacard at this location and the
    original datacard remains unchanged. When the datacard contains no shape block, the context
    manager yields an empty list. If this list is filled by the calling frame, a new shape block is
    added with these lines upon closure.
    """
    # read the datacard content in blocks
    datacard = os.path.abspath(os.path.expandvars(os.path.expanduser(datacard)))
    blocks = read_datacard_blocks(datacard)

    # the datacard contains shapes when the first line of the second block starts with "shapes "
    had_shapes = len(blocks) >= 2 and blocks[1][0].startswith("shapes ")

    # get shape lines
    shape_lines = blocks[1] if had_shapes else []

    # yield shape lines and keep track of changes via hashes
    hash_before = law.util.create_hash(shape_lines)
    yield shape_lines
    hash_after = law.util.create_hash(shape_lines)
    has_changes = hash_after != hash_before

    # prepare the target location when given
    if target_datacard:
        target_datacard = os.path.abspath(os.path.expandvars(os.path.expanduser(target_datacard)))
        target_dirname = os.path.dirname(target_datacard)
        if not os.path.exists(target_dirname):
            os.makedirs(target_dirname)

    # handle saving of changes
    if has_changes:
        # inject shape lines into block of there were none before
        # or remove the shape block if there was one, but there are no shape lines left
        if shape_lines and not had_shapes:
            blocks.insert(1, shape_lines)
        elif not shape_lines and had_shapes:
            blocks.pop(1)

        # write the new lines
        with open(target_datacard or datacard, "w") as f:
            block_sep = "\n" + 80 * "-" + "\n"
            f.write(block_sep.join("\n".join(block) for block in blocks))

    elif target_datacard:
        # no changes, just copy the original file
        shutil.copy2(datacard, target_datacard)


def extract_shape_files(datacard, absolute=True, resolve=True):
    """
    Extracts all unique paths declared as shape files in a *datacard* and returns them in a list.
    When *absolute* is *True*, the extracted paths are made absolute based on the location of the
    datacard itself. When both *absolute* and *resolve* are *True*, symbolic links are resolved.
    """
    # read shape lines and extract file paths
    shape_files = []
    with manipulate_shape_lines(datacard) as shape_lines:
        for line in shape_lines:
            # the shape file is the 4th part
            parts = line.split()
            if len(parts) >= 4:
                shape_files.append(parts[3])

    # convert to absolute paths (when paths are already absolute, os.path.join will not change them)
    if absolute:
        dirname = os.path.dirname(datacard)
        shape_files = [os.path.join(dirname, f) for f in shape_files]

        if resolve:
            shape_files = [os.path.realpath(f) for f in shape_files]

    # make them unique
    shape_files = law.util.make_unique(shape_files)

    return shape_files


def update_shape_files(func, datacard, target_datacard=None):
    """
    Updates the shape files in a *datacard* according to a configurable function *func* that should
    accept the shape file location, the process name, the channel name and optionally the nominal
    and systematic histogram extraction patterns that are usually defined in each shape line. When
    a *target_datacard* is given, the updated datacard is stored at this path rather than the
    original one. Example:

    .. code-block:: python

        def func(shape_file, process, channel, *patterns):
            return shape_file.replace(".root", "_new.root")

        update_shape_files(func, "datacard.txt")
    """
    # use the shape line manipulation helper
    with manipulate_shape_lines(datacard, target_datacard=target_datacard) as shape_lines:
        # iterate through shape lines and change them in-place
        for i, line in enumerate(shape_lines):
            parts = line.split()
            if len(parts) < 4:
                continue

            # extract fields
            prefix = parts.pop(0)  # usually "shapes"
            process = parts.pop(0)
            channel = parts.pop(0)
            shape_file = parts.pop(0)
            patterns = parts

            # run the func
            new_shape_file = func(shape_file, process, channel, *patterns)

            # update if needed
            if new_shape_file != shape_file:
                new_line = " ".join([prefix, process, channel, new_shape_file] + patterns)
                shape_lines[i] = new_line
