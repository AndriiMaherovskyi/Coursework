# Coursework

Coursework created by Andrii Mherovskyi and Yuriy Bondar.
The target of this code is to group together countries with most common indicators.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install virtual environment(venv).

```bash
pip install virtualenv
```
Then you have to create a new virtual environment for the project. For example we call it "venv".

```bash
virtualenv venv
```

And finally activate it by the next command.

```bash
venv/Scripts/activate
```
## Set up needed libs

The next step is to set up pycuda lib.

```bash
pip install pycuda
```

## Occasion situation !!!

It is quite possible that you'll need to set up some path in your operation system. In Windows you can solve it added the following paths:

```bash
C:\Program Files (x86)\Microsoft Visual Studio\<your version>\Community\VC\Tools\MSVC\<version>\bin\Hostx64\x64
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\<version>\bin
```
Or you have to add this paths, it's depends where your Visual studio located.

```bash
C:\Program Files\Microsoft Visual Studio\<your version>\Community\VC\Tools\MSVC\<version>\bin\Hostx64\x64
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\<version>\bin
```
