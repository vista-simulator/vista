.. vista documentation master file, created by
   sphinx-quickstart on Mon Oct  4 15:17:55 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to VISTA's documentation!
=================================



.. toctree::
   :maxdepth: 1

   Introduction <introduction/index.rst>
   Getting Started <getting_started/index.rst>
   Advanced Usage <advanced_usage/index.rst>
   Interface To Public Datasets <interface_to_public_dataset/index.rst>
   API Documentation <api_documentation/index.rst>
   FAQ <faq/index.rst>
   Acknowledgement <acknowledgement/index.rst>



.. image:: /_static/vista_logo.gif
   :width: 300

`VISTA <https://vista.csail.mit.edu>`_ is a data-driven simulation engine for autonomous driving perception and control. The VISTA API provides an interface for transforming real-world datasets into virtual environments with dynamic agents, sensor suites, and task objectives.

All code is available on our `github repository <https://github.com/vista-simulator/vista>`_.


Installation
++++++++++++
VISTA can be installed into your python 3 environment using the `PyPi package interface <https://pypi.org/project/vista/>`_. 

::

    >> pip install vista

Please also ensure that you have all required dependencies to successfully run VISTA. Details on dependencies are outlined in :ref:`getting_started-installation`.



|paper1| |paper2| |paper3|



.. |paper1| image:: /_static/paper1.jpg
  :target: https://ieeexplore.ieee.org/document/8957584
  :width: 31%

.. |paper2| image:: /_static/paper2.jpg
  :target: https://arxiv.org/abs/2111.12083
  :width: 31%

.. |paper3| image:: /_static/paper3.jpg
  :target: https://arxiv.org/abs/2111.12137
  :width: 31%



Citing VISTA
++++++++++++

.. code-block::

    % VISTA 1.0: Sim-to-real RL
    @article{amini2020learning,
       title={Learning Robust Control Policies for End-to-End Autonomous Driving from Data-Driven Simulation},
       author={Amini, Alexander and Gilitschenski, Igor and Phillips, Jacob and Moseyko, Julia and Banerjee, Rohan and Karaman, Sertac and Rus, Daniela},
       journal={IEEE Robotics and Automation Letters},
       year={2020},
       publisher={IEEE}
    }

    % VISTA 2.0: Multi-sensor simulation
    @inproceedings{amini2022vista,
     title={VISTA 2.0: An Open, Data-driven Simulator for Multimodal Sensing and Policy Learning for Autonomous Vehicles},
     author={Amini, Alexander and Wang, Tsun-Hsuan and Gilitschenski, Igor and Schwarting, Wilko and Liu, Zhijian and Han, Song and Karaman, Sertac and Rus, Daniela},
     booktitle={2022 International Conference on Robotics and Automation (ICRA)},
     year={2022},
     organization={IEEE}
    }

    % VISTA 2.0: Multi-agent simulation
    @inproceedings{wang2022learning,
     title={Learning Interactive Driving Policies via Data-driven Simulation},
     author={Wang, Tsun-Hsuan and Amini, Alexander and Schwarting, Wilko and Gilitschenski, Igor and Karaman, Sertac and Rus, Daniela},
     booktitle={2022 International Conference on Robotics and Automation (ICRA)},
     year={2022},
     organization={IEEE}
    }


Contribution Guidelines
+++++++++++++++++++++++

VISTA is constantly being advanced and has been built with research, extensibility, and community development as a priority. We actively encourage contributions to the VISTA repository and codebase, including issues, enhancements, and pull requests.
