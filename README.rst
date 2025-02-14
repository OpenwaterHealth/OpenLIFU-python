openlifu
========

|Actions Status| |Documentation Status|

|PyPI version| |PyPI platforms|

|GitHub Discussion|

.. SPHINX-START

``openlifu`` is a toolbox for planning and controlling focused
ultrasound treatments. It generally replicates the functionality
developed in the MATLAB `open-TFUS
toolbox <https://github.com/OpenwaterHealth/opw_neuromod_sw>`__.

Installation
------------

Requirements
~~~~~~~~~~~~

Python 3.9 or later on Windows or Linux.

Create Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Windows:

.. code:: sh

   C:\Users\<uname>\AppData\Local\Programs\Python\Python311\python.exe -m venv env
   .\env\Scripts\activate

Linux:

.. code:: sh

   python3.11 -m venv env

Install project (editable)
~~~~~~~~~~~~~~~~~~~~~~~~~~

With this repo as the working directory:

Basic mode
^^^^^^^^^^
.. code:: sh

   pip install -e .

Dev mode
^^^^^^^^
.. code:: sh

   pip install -e '.[dev]'

Version control of database using DVC (Data Version Control)
-------------------------------------------------------------

Data Version Control (DVC) is a data management tool that is meant to be run alongside Git.
In this project, DVC is used to link changes in the code to specific versions of a sample database containing example project files.
DVC can be used when this project is installed in Dev mode. You can read more about DVC and how to use it `here <https://dvc.org/doc/start>`_.
**Note:** Remote access to the sample database stored on google drive is currently restricted. Access requires a :code:`gdrive_client_secret`
for user access authentication to be shared by developers.

DVC usage
~~~~~~~~~

To download the sample database:

.. code:: sh

   git pull
   dvc remote modify --local shared_gdrive gdrive_client_secret <client_secret_here> # Contact developers for grive_client_secret
   dvc pull # Requires access to remote storage

This will download a directory 'db_dvc' in the repo directory that
contains the corresponding version of example database files.

To commit updates to the sample database:

.. code:: sh

   dvc add db_dvc
   git commit -m "Describe updates to database"
   git push
   dvc push #Requires access to remote storage

Set Python Path
.. code:: sh
   set PYTHONPATH=%cd%\src

Disclaimer
----------

CAUTION - Investigational device. Limited by Federal (or United States)
law to investigational use. The system described here has *not* been
evaluated by the FDA and is not designed for the treatment or diagnosis
of any disease. It is provided AS-IS, with no warranties. User assumes
all liability and responsibility for identifying and mitigating risks
associated with using this software.

.. |Actions Status| image:: https://github.com/OpenwaterHealth/OpenLIFU-python/workflows/CI/badge.svg
   :target: https://github.com/OpenwaterHealth/OpenLIFU-python/actions
.. |Documentation Status| image:: https://readthedocs.org/projects/openlifu/badge/?version=latest
   :target: https://openlifu.readthedocs.io/en/latest/?badge=latest
.. |PyPI version| image:: https://img.shields.io/pypi/v/openlifu
   :target: https://pypi.org/project/openlifu/
.. |PyPI platforms| image:: https://img.shields.io/pypi/pyversions/openlifu
   :target: https://pypi.org/project/openlifu/
.. |GitHub Discussion| image:: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
   :target: https://github.com/OpenwaterHealth/OpenLIFU-python/discussions
