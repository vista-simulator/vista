.. _getting_started-installation:

Installation
============

1. Install system dependencies. ::

    >> sudo apt-get update
    >> sudo apt-get install -y freeglut3-dev
    >> sudo apt-get install -y libglib2.0-0
    >> sudo apt-get install -y ffmpeg

2. [Option A] Create a Conda environment using the provided ``environment.yaml``. ::

    >> conda create -n vista python=3.8
    >> conda env update --name vista --file ./environment.yaml

3. [Option B] Install the following python dependencies.

    * `opencv-python <https://github.com/opencv/opencv-python>`_
    * `ffio <https://pypi.org/project/ffio/>`_
    * `shapely <https://github.com/Toblerity/Shapely>`_
    * `descartes <https://pypi.org/project/descartes/>`_
    * `matplotlib <https://matplotlib.org/>`_
    * `pyrender <https://github.com/mmatl/pyrender>`_
    * `torch <https://github.com/pytorch/pytorch>`_
    * `torchvision <https://github.com/pytorch/vision>`_
    * `pickle5 <https://github.com/pitrou/pickle5-backport>`_
    * `h5py <https://github.com/h5py/h5py>`_


4. Install VISTA from `PyPi <https://pypi.org/project/vista/>`_.::

    >> pip install vista


5. Try to import Vista. ::

    >> python -c "import vista"

6. Download and extract sample dataset from `here <https://www.dropbox.com/s/62pao4mipyzk3xu/vista_traces.zip?dl=1>`_. For more details, check :doc:`Data Format <./data_format>`.

.. note::

  Dataset above is a small sample dataset to test functionality. Much more data and environments to come!
