# Dockerfile for QuickZonoReach

FROM python:3.6

# install other (required) dependencies
RUN pip3 install numpy scipy matplotlib

# copy current folder to docker
COPY . /quickzonoreach

# set environment variable
ENV PYTHONPATH=$PYTHONPATH:/quickzonoreach

### As default command: run the tests ###
CMD python3 /quickzonoreach/examples/example_plot.py && python3 /quickzonoreach/examples/example_compare.py && python3 /quickzonoreach/examples/example_time_varying.py && python3 /quickzonoreach/examples/example_profile.py

# USAGE:
# Build container and name it 'quickzonoreach':
# docker build . -t quickzonoreach

# # run example scripts (default command)
# docker run quickzonoreach

# # get a shell:
# docker run -it quickzonoreach bash
# to delete docker container use: docker rm quickzonoreach
