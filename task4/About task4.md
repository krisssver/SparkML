####  Файлы
- PySparkMLFit.py - процесс обучения модели с оптимизацией гиперпараметров <br /> 
*python PySparkMLFit.py --data_path=session-stat.parquet --model_path=spark_ml_model*
- PySparkMLPredict.py - прогноз на данных и его сохранение <br /> 
*python PySparkMLPredict.py --data_path=test.parquet --model_path=spark_ml_model --result_path=result*

Предоставленные датасеты:
1) session-stat.parquet - файл с данным со статистикой по сессиям
2) test.parquet - файл с данными со статистикой по сессиям, для оценки вашей модели (не имеет целевого признака is_bot)
