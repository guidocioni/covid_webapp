FROM continuumio/miniconda3:latest

# get latest fixes for conda
RUN /opt/conda/bin/conda update conda -y

# Disable Intel optimizations (takes a lot of extra space). (tnx kennethreitz)
RUN /opt/conda/bin/conda install nomkl -y

# Install desired packages
RUN /opt/conda/bin/conda install -y scipy dash Flask gunicorn numpy pandas plotly Werkzeug flask-caching beautifulsoup4 xlrd

# Cleanup conda files
RUN /opt/conda/bin/conda clean -a -y

# Add conda to path
ENV PATH /opt/conda/bin:$PATH

# Add our code
ADD ./ /opt/webapp/
WORKDIR /opt/webapp

CMD /opt/conda/bin/gunicorn --timeout 90 app:server
