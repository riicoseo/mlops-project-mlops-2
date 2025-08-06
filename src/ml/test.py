from src.api import get_mlflow_model

def test():    
    model = get_mlflow_model()
    print(model.metadata)

def test2():
    from src.api import mlflow_model
    print(mlflow_model.metadata)


if __name__ == "__main__":
    test()
    test2()
