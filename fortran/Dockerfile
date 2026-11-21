# Reference build of XBeach (SVN trunk) for the WebGPU verification pipeline.
# Build context must be the trunk/ directory:
#   docker build -t xbeach-ref -f fortran/Dockerfile trunk
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran gcc make autoconf automake libtool pkg-config \
    libnetcdf-dev libnetcdff-dev \
    python3 python3-mako \
    subversion ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/xbeach
WORKDIR /opt/xbeach

# python -> python3 for scripts/generate.py shebang
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# Pre-generate mako .inc files: the Makefile lacks proper deps on them, which
# breaks parallel (and even first-pass) builds.
RUN ./autogen.sh && \
    FCFLAGS="-I/usr/include -O2" ./configure --with-netcdf --without-mpi && \
    (cd src/xbeachlibrary && ../../scripts/generate.py) && \
    make -j"$(nproc)" || make && \
    make install && \
    ldconfig

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnetcdff7t64 libnetcdf19t64 libgfortran5 netcdf-bin \
    && rm -rf /var/lib/apt/lists/* || \
    (apt-get update && apt-get install -y --no-install-recommends \
     libnetcdff7 libnetcdf19 libgfortran5 netcdf-bin \
     && rm -rf /var/lib/apt/lists/*)
COPY --from=builder /usr/local/ /usr/local/
RUN ldconfig
WORKDIR /data
ENTRYPOINT ["/usr/local/bin/xbeach"]
