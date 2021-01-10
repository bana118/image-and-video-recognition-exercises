# image-and-video-recognition-exercises


# jupyterlab

```
jupyter lab --port=8080 --ip=0.0.0.0 --no-browser --allow-root
```

# exercise2
MNIST example  
- pytorch  
    [examples/mnist at master · pytorch/examples](https://github.com/pytorch/examples/tree/0f0c9131ca5c79d1332dce1f4c06fe942fbdc665/mnist)
    ```
    python src/exercise2/pytorch.py --no-cuda
    ```
    Test set: Average loss: 0.0270, Accuracy: 9911/10000 (99%)
- tensorflow  
    [docs\-l10n/beginner\.ipynb at master · tensorflow/docs\-l10n](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tutorials/quickstart/beginner.ipynb)
    ```
    jupyter lab --port=8080 --ip=0.0.0.0 --no-browser --allow-root
    ```
    313/313 - 0s - loss: 0.0804 - accuracy: 0.9763

# final
## model search

Windows
```
python .\src\final\cnn_reg_search.py > .\src\final\output\00all.log 2>&1
```
