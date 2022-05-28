# WhileTrue_walRUSfinder
With this also one can detect walruses in the wild from drone shots

----

## To run our best solution

```pip install -r requirements.txt```

```python inferenceSEGM.py ./models/unet.pth /path/to/images/ /output/path/

It generates you processed images, masks and walrus center coordinates in csv files in folder /output/path_csv/
