# classification_python
Лабораторная №2 вариант лабораторной из задачника №3  по курсу Анализ информационных технологий (ИТ-2 МО)
Задачник: https://moodle.uniyar.ac.ru/pluginfile.php/321959/mod_resource/content/0/python_labs_classification.pdf
лог:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 keras_layer (KerasLayer)    (None, 1024)              93600852  
                                                                 
 dense (Dense)               (None, 256)               262400    
                                                                 
 dense_1 (Dense)             (None, 1)                 257       
                                                                 
=================================================================
Total params: 93,863,509
Trainable params: 262,657
Non-trainable params: 93,600,852
_________________________________________________________________
Train on 2228 samples, validate on 1671 samples

64/2228 [..............................] - ETA: 3:35 - loss: 0.6968 - accuracy: 0.4688
128/2228 [>.............................] - ETA: 5:41 - loss: 0.5201 - accuracy: 0.6875
192/2228 [=>............................] - ETA: 4:45 - loss: 0.4639 - accuracy: 0.7500
2228/2228 [==============================] - ETA: 0s - loss: 0.1490 - accuracy: 0.9336
2228/2228 [==============================] - 619s 278ms/
sample - loss: 0.1490 - accuracy: 0.9336 - val_loss: 0.0749 - val_accuracy: 0.9797
precision    recall  f1-score   support

       rumor       0.98      0.99      0.99      1445
 not a rumor       0.96      0.89      0.92       228

    accuracy                           0.98      1673
   macro avg       0.97      0.94      0.96      1673
weighted avg       0.98      0.98      0.98      1673


Process finished with exit code 0

### отчет распознавания
![alt text](https://github.com/aiserrock/classification_python/blob/main/log2.jpg) 

### Зависимость уменьшения ошибок распознавания, от продолжения обучения  
![alt text](https://github.com/aiserrock/classification_python/blob/main/epoch_losses.jpg) 
### Зависимость повышения точности распознавания, от продолжения обучения  
![alt text](https://github.com/aiserrock/classification_python/blob/main/epoch_acc.jpg) 
