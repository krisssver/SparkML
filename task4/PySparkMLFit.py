import operator
import argparse

from pyspark.ml.pipeline import PipelineModel
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

MODEL_PATH = 'spark_ml_model'
LABEL_COL = 'is_bot'


def process(spark, data_path, model_path):
    """
    Основной процесс задачи.

    :param spark: SparkSession
    :param data_path: путь до датасета
    :param model_path: путь сохранения обученной модели
    """
    #prepare_data
    print('-----prepare data-----')
    data = spark.read.parquet(data_path)
    train, test = data.randomSplit([0.8, 0.2])
    user_type_index = StringIndexer(inputCol='user_type', outputCol="user_type_index")
    platform_index = StringIndexer(inputCol='platform', outputCol="platform_index")
    feature = VectorAssembler(
    inputCols=["user_type_index","duration","platform_index","item_info_events","select_item_events","make_order_events","events_per_min"],
    outputCol="features")
    
    #pipeline and save model
    print('-----pipeline and save model-----')
    rf_classifier = RandomForestClassifier(labelCol=LABEL_COL, featuresCol="features")
    pipeline = Pipeline(stages=[user_type_index, platform_index, feature, rf_classifier])
    p_model = pipeline.fit(train)
    print(type(p_model))
    p_model.write().overwrite().save('p_model')
    
    #tuning
    print('-----tuning-----')
    model = PipelineModel.load('p_model')
    evaluator = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="accuracy")
    paramGrid = ParamGridBuilder()\
                  .addGrid(rf_classifier.maxDepth, [2, 3, 4])\
                  .addGrid(rf_classifier.maxBins, [4, 5, 6])\
                  .addGrid(rf_classifier.minInfoGain, [0.05, 0.1, 0.15])\
                  .build()
    tvs = TrainValidationSplit(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            trainRatio=0.8)
    model = tvs.fit(train)
    
    #best model
    print('-----best model-----')
    jo = model.bestModel.stages[-1]._java_obj
    print('Max Depth: {}'.format(jo.getMaxDepth()))
    print('Num Trees: {}'.format(jo.getMaxBins()))
    print('Impurity: {}'.format(jo.getMinInfoGain()))
    rf_classifier = RandomForestClassifier(labelCol=LABEL_COL, featuresCol="features", maxDepth=jo.getMaxDepth(), maxBins=jo.getMaxBins(), minInfoGain=jo.getMinInfoGain())
    p_model = pipeline.fit(train)
    p_model.write().overwrite().save(model_path)
    
    print('-----Finished OK-----')

def main(data_path, model_path):
    spark = _spark_session()
    process(spark, data_path, model_path)


def _spark_session():
    """
    Создание SparkSession.

    :return: SparkSession
    """
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='session-stat.parquet', help='Please set datasets path.')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Please set model path.')
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    main(data_path, model_path)
