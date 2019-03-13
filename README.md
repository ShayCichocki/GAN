# Simple GAN to be used as a base for other projects, based on Udacity GAN lectures
## Purpose
For getting into creating new images with GAN, I found it easiest to abstract out a simple GAN and fiddle with the code

## 1. Installation

**Download** the latest version of `miniconda` that matches your system.

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

## 2. Create and Activate the Environment

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac/Linux, a normal terminal window will work. 

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/ShayCichocki/GAN
cd GAN
```

2. Create (and activate) a new environment, named `gan-playground` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -f environment.yml
	source activate gan-playground
	```
	- __Windows__: 
	```
	conda create -f environment.yml python=3.6
	activate gan-playground
	```
## 3. Add your data and run the program
1. For this project to work correctly, the data needs to be structured with subfolders for example:
    ```
    -data-set
        |-train 	
            |-your images here
    ```
    You can configure the dataset location to be anywhere in the `.env`
2. After adding your relevant images to the train folder, copy `.env.example` into `.env` and fiddle with training params
3. Run `python runner.py` from your activated conda environment
4. This will print out each loss from the Discriminator and Generator
5. After the training is over your results will be displayed in a libplot window

## TODOS
1. Add saving/loading of Discriminator/Generator
2. Cleanup code