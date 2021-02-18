FROM python:3.8

# ports
EXPOSE 8000

# minimal software stack
RUN apt-get update; apt-get clean
RUN apt-get install -y nano less htop git sshpass; apt-get clean

# update pip
RUN pip install --no-cache-dir --upgrade pip

# python software stack
RUN pip install --no-cache-dir mkdocs
RUN pip install --no-cache-dir mkdocs-material
RUN pip install --no-cache-dir mkdocs-markdownextradata-plugin
RUN pip install --no-cache-dir mkdocs-git-revision-date-localized-plugin
RUN pip install --no-cache-dir mkdocs-awesome-pages-plugin
RUN pip install --no-cache-dir mkdocs-minify-plugin

# change the workdir to /documentation
# the inference repository is not public and should be cloned locally
# with user permissions and then mounted into this directory
WORKDIR /documentation

# add documentation extensions to the PYTHONPATH
ENV PYTHONPATH "/documentation/docs:$PYTHONPATH"

# default command
CMD ["mkdocs", "build", "--strict"]
