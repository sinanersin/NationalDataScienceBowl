### This project was made during MSc. Data Science studies in University of Amsterdam.

It is made by me (Sinan Ersin) and Jan Jetze Beitler.

The project was a small-scale project of the real National Data Science Bowl. 

I cold not find the datasets, though the code is more important.

In the file model.py the final model can be found. The other .py files just contain helper functions.

The code in the .ipynb file is pretty straight-forward. In the 'Globals' part, the global settings are defined, like the directory of the data etc.
In the Globals block you can also change the settings like the image size or the number of epochs used.

Once the data directories are set, the scripts can be runned at once, but this is not recommended.
The first three blocks of code (Imports, Globals and Loading data) are necessary to run the script as well as the first block of part 3 (Models).
Then it can be chosen to do either a normal run or a test using cross_validation. Once a normal run is done, it is also possible to make a submission. This is not possible when only tests are done.

For explanation about the 'parameters', which is used for the cross_validation, see the markdown below the block of code in the notebook.
