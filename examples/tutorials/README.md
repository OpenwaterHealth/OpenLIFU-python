These notebooks demonstrate the functionality and usage of openlifu.

# Using the jupytext notebooks

The notebooks are in a [jupytext](https://jupytext.readthedocs.io/en/latest/)
format. To use them, first install jupyter notebook and jupytext to the python
environment in which openlifu was installed:

```sh
pip install notebook jupytext
```

Then, either

- launch a jupyter notebook server and choose to open the notebook `.py` files
  with jupytext (right click -> open with -> jupytext notebook), or
- create paired `.ipynb` files to be used with the notebooks and then use the
  `.ipynb` files as normal:
  ```sh
  jupytext --to ipynb *.py
  ```

The paired `.ipynb` files will automatically be kept in sync with the `.py`
files, so the `.py` files can be used in version control and the `.ipynb` files
never need to be committed.
