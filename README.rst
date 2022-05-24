VISTA Autonomous Driving Simulator
==================================

.. image:: /_static/overview.png

`VISTA <https://vista.csail.mit.edu>`_ is a data-driven simulation engine for autonomous driving perception and control. The VISTA API provides an interface for transforming real-world datasets into virtual environments with dynamic agents, sensor suites, and task objectives.


Installation
++++++++++++
VISTA can be installed into your Python 3 environment using the `PyPi package interface <https://pypi.org/project/vista/>`_.

::

    >> pip install vista

Please also ensure that you have all required dependencies to successfully run VISTA. Details on dependencies are outlined in :ref:`getting_started-installation`.


|paper1| |paper2| |paper3|

.. |paper1| image:: docs/_static/paper1.png
   :width: 31%

.. |paper2| image:: docs/_static/paper2.png
   :width: 31%

.. |paper3| image:: docs/_static/paper3.png
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
