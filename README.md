# WhileTrue_walRUSfinder
With this also one can detect walruses in the wild from drone shots


в папке results ответы на тестовый датасет(2 варианта, но по нашему мнению от Unet получился получше)

запускать все инеференсы через командную строку, указав 3 аргумента:
1. путь к модели
2. путь к папке с фото
3. название папки с результатом(создавать необязательно)

предварительно установите необходимые библиотеки, описанные в txt файле

```pip install -r requirements.txt```

```python inferenceSEGM.py ./models/unet.pth /path/to/images/ /output/path/```

zip архив models скачать и распаковать распаковать, если не скачались модели


https://drive.google.com/file/d/1wC0B5-q6NQYWkQFgLAuSEECWgeCOaB2T/view?usp=sharing
