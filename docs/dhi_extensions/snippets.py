# coding: utf-8

"""
Custom snippets extension that supports line slicing.
"""

from markdown import Extension
from markdown.preprocessors import Preprocessor
import re
import codecs
import os


class DHISnippetPreprocessor(Preprocessor):

    RE_ALL_SNIPPETS = re.compile(
        r'''(?x)
        ^(?P<space>[ \t]*)
        (?P<all>
            (?P<inline_marker>-{2,}8<-{2,}[ \t]+)
            (?P<snippet>(?:"(?:\\"|[^"\n\r])+?"|'(?:\\'|[^'\n\r])+?'))(?![ \t]) |
            (?P<block_marker>-{2,}8<-{2,})(?![ \t])
        )\r?$
        '''
    )

    RE_SNIPPET = re.compile(
        r'''(?x)
        ^(?P<space>[ \t]*)
        (?P<snippet>.*?)\r?$
        '''
    )

    def __init__(self, config, md):
        self.base_path = config.get('base_path')
        self.encoding = config.get('encoding')
        self.check_paths = config.get('check_paths')
        self.tab_length = md.tab_length

        super(DHISnippetPreprocessor, self).__init__()

    def split_path(self, path):
        if "@" not in path:
            return path, None

        path, num_str = path.split("@", 1)

        def try_int(s):
            try:
                return int(s)
            except:
                raise ValueError("cannot convert snippet line slice argument to int: {}".format(s))

        line_nums = []
        for s in num_str.strip().split(","):
            s = s.strip()
            if "-" in s:
                start, stop = s.split("-", 1)
                start = start and try_int(start) or 1
                stop = stop and try_int(stop) or int(1e7)
                line_nums.extend(range(start, stop + 1))
            else:
                line_nums.append(try_int(s))

        return path, line_nums

    def parse_snippets(self, lines, file_name=None):
        new_lines = []
        inline = False
        block = False
        for line in lines:
            inline = False
            m = self.RE_ALL_SNIPPETS.match(line)
            if m:
                if block and m.group('inline_marker'):
                    # Don't use inline notation directly under a block.
                    # It's okay if inline is used again in sub file though.
                    continue
                elif m.group('inline_marker'):
                    # Inline
                    inline = True
                else:
                    # Block
                    block = not block
                    continue
            elif not block:
                # Not in snippet, and we didn't find an inline,
                # so just a normal line
                new_lines.append(line)
                continue

            if block and not inline:
                # We are in a block and we didn't just find a nested inline
                # So check if a block path
                m = self.RE_SNIPPET.match(line)

            if m:
                # Get spaces and snippet path.  Remove quotes if inline.
                space = m.group('space').expandtabs(self.tab_length)
                path = m.group('snippet')[1:-1].strip() if inline else m.group('snippet').strip()

                if not inline:
                    # Block path handling
                    if not path:
                        # Empty path line, insert a blank line
                        new_lines.append('')
                        continue
                if path.startswith('; '):
                    # path stats with '#', consider it commented out.
                    # We just removing the line.
                    continue

                snippet = os.path.join(self.base_path, path)
                if snippet:
                    snippet, line_nums = self.split_path(snippet)
                    if os.path.exists(snippet):
                        if snippet in self.seen:
                            # This is in the stack and we don't want an infinite loop!
                            continue
                        if file_name:
                            # Track this file.
                            self.seen.add(file_name)
                        try:
                            with codecs.open(snippet, 'r', encoding=self.encoding) as f:
                                file_lines = f.readlines()
                                if line_nums:
                                    file_lines = {i + 1: l for i, l in enumerate(file_lines)}
                                    file_lines = [file_lines[n] for n in line_nums if n in file_lines]
                                new_lines.extend(
                                    [space + l2 for l2 in self.parse_snippets([l.rstrip('\r\n') for l in file_lines], snippet)]
                                )
                        except Exception:  # pragma: no cover
                            pass
                        if file_name:
                            self.seen.remove(file_name)
                    elif self.check_paths:
                        raise IOError("Snippet at path %s could not be found" % path)

        return new_lines

    def run(self, lines):
        self.seen = set()
        return self.parse_snippets(lines)


class DHISnippetExtension(Extension):

    def __init__(self, *args, **kwargs):
        self.config = {
            'base_path': [".", "Base path for snippet paths - Default: \"\""],
            'encoding': ["utf-8", "Encoding of snippets - Default: \"utf-8\""],
            'check_paths': [False, "Make the build fail if a snippet can't be found - Default: \"false\""]
        }

        super(DHISnippetExtension, self).__init__(*args, **kwargs)

    def extendMarkdown(self, md):
        self.md = md
        md.registerExtension(self)
        config = self.getConfigs()
        dhi_snippet = DHISnippetPreprocessor(config, md)
        md.preprocessors.register(dhi_snippet, "dhi_snippet", 32)


def makeExtension(*args, **kwargs):
    return DHISnippetExtension(*args, **kwargs)
