# ArtStyleTransfer

This is an implementation of paper "A Neural Algorithm of Artistic Style".
https://arxiv.org/pdf/1508.06576.pdf for reference.

This is the final project for Columbia University STAT GR5242 Advance Machine Learning.

Group Leader:Haiqi Li

Team Member:Yutong Zhang,Yifan Wu,Ziyan Xu

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

## Command detail

To run the code, just cd into the directory and run the following code:

```sh
python ArtStyleTransfer.py --content content.jpg --style style.jpg --output output.jpg
```

--iter:num of iterations you want to specify.Default 400.

```sh
python ArtStyleTransfer.py --content content.jpg --style style.jpg --output output.jpg --iter 600
```

--record:Default False. True for record the loss of each step and plot them in output dir.

```sh
python ArtStyleTransfer.py --content content.jpg --style style.jpg --output output.jpg --record T
```

--flw:The weight of each feature layer in VGG structure.The exact weight is in Settings.py.The number of it is the index of layer weight list.Default 0 to be [.25,.25,.25,.25]

```sh
python ArtStyleTransfer.py --content content.jpg --style style.jpg --output output.jpg --flw 3
```

--lt:loss type.Default to be sqaure loss.Another choice is absolute loss.Note the loss function is changed.

```sh
python ArtStyleTransfer.py --content content.jpg --style style.jpg --output output.jpg --lt AE
```

--rstep:record per step.Default 50 means record target picture every 50 steps.

```sh
python ArtStyleTransfer.py --content content.jpg --style style.jpg --output output.jpg --rstep 10
```

--alpha --beta:The parameter in paper of loss weight.Alpha is weight of content loss and beta is weight of style loss.

```sh
python ArtStyleTransfer.py --content content.jpg --style style.jpg --output output.jpg --alpha 10.0
```

--fromc:The target picture initialization method.Default False to be random initialization.True to initialize from content picture.

```sh
python ArtStyleTransfer.py --content content.jpg --style style.jpg --output output.jpg --fromc T
```

--cont:Continue training.Recommend not to use this.

If you really want to try this,go as the following:

1.upload the sList.dat from output to dir output

2.Specify iteration to be like 

iteration=iteration you have already trained + iteration you want to go further

EX:if you have trained 400 iterations and want to continue to 600 iterations as total,then

```sh
python ArtStyleTransfer.py --content content.jpg --style style.jpg --output output.jpg --cont T --iter 600
```

It will train another 200 iterations.



