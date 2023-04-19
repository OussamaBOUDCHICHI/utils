
---

:warning:  <span style="color:darkblue"> **Make sure to install gsl libray, if it's not installed. Link for windows users : [gsl windows](https://www.gnu.org/software/gsl/). For mac users, if you have MacPorts just type the following command in terminal:** </span>

```
sudo port install gsl 
```
<span style="color:darkblue"> **else, install homebrew, and type:** </span>

```
brew install gsl
```

---
:warning:  <span style="color:darkred"> **Please modify gsl_include/library paths in setup.py to the paths in your device.** </span>

--- 

do : 
```
cd path/to/CODE
make
```
else, for Windows users who don't have a bash/zsh terminal:

```python
cd path/to/CODE
python3.11 setup.py build_ext --inplace
python3.11 -m unittest test.py
	
```

