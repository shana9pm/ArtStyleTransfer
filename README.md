# ArtStyleTransfer

This is an implementation of paper "A Neural Algorithm of Artistic Style".
https://arxiv.org/pdf/1508.06576.pdf for reference.

This should be the final project for Columbia University STAT GR5242 Advance Machine Learning.

This is the raw version of markdown that I will add other features and some project paper someday.

# File structure

```
project/
   |---- input/
   |       |---- content/
   |       |        |---- content.jpg
   |       |---- style/
   |       |        |---- style.jpg
   |---- output/
   |       |---- output.jpg
   |
   |---- ArtStyleTransfer.py
   |----Settings.py	
   |----utils.py
```

To run the code, just cd into the directory and run the following code:

```sh
python ArtStyleTransfer.py --content content.jpg --style style.jpg --output output.jpg
```

Default iteration is 600. If you want to specify iterarions:

```sh
python ArtStyleTransfer.py --content content.jpg --style style.jpg --output output.jpg --iter 600
```