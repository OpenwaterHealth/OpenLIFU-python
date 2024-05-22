****
Home
****

Open-pyFUS is a toolbox for planning and controlling focused ultrasound treatments. It generally replicates the functionality developed in the MATLAB `open-TFUS toolbox <https://github.com/OpenwaterHealth/opw_neuromod_sw>`_.

Installation
------------

Requirements
^^^^^^^^^^^^
Python 3.11
USTX v2 
USB Type-C cable
12VDC Power Supply Barrel Jack
High Voltage Power Supply (HVPS)    

Power Supply Connection
^^^^^^^^^^^^^^^^^^^^^^^
12VDC Power Supply Barrel Jack connect to bottom board
HVPS connect to top board MOLEX connector Red Wire = -HV, Black Wire = Common Ground, Yellow Wire = +HV

Create Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^
``cd open_pyfus``
``C:\Users\<uname>\AppData\Local\Programs\Python\Python311\python.exe -m venv env``

Install Requirements
^^^^^^^^^^^^^^^^^^^^
``.\env\Scripts\activate``

``pip install --upgrade pip setuptools``
``pip install -r requirements.txt``



Run Demo Application
^^^^^^^^^^^^^^^^^^^^
NOTE: Pin mapping for the demo application can be changed by modifying pinmap.json file

``python DemoApplication.py``


Run Notebook
^^^^^^^^^^^^^^^^^^^^
``jupyter notebook``

``open_pyfus\Demo_Notebook.ipynb``

Disclaimer
----------
CAUTION - Investigational device. Limited by Federal (or United States) law to investigational use. The system described here has *not* been evaluated by the FDA and is not designed for the treatment or diagnosis of any disease. It is provided AS-IS, with no warranties. User assumes all liability and responsibility for identifying and mitigating risks associated with using this software.
