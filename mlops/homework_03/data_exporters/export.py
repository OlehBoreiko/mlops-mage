import mlflow
#import mlflow.sklearn
import pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data(data, *args, **kwargs):

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment('ob-yellow-taxi-march1')

    dv, model = data

    with mlflow.start_run(nested=True):
        mlflow.sklearn.log_model(model, artifact_path="models")
        with open("dv.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact('dv.pkl', 'dv')
