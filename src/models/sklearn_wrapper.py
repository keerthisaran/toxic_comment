from sklearn.base import BaseEstimator, ClassifierMixin

class TF_to_SK(BaseEstimator,ClassifierMixin):
    
    def __init__(self,model,init_params,compile_params,batch_size=64,epochs=10):
        # self.model=compiled_tf_model
        self.batch_size=batch_size
        self.epochs=epochs
        self.model=self.get_tf_model(model,init_params,compile_params)
    

    
    def compile(self,loss,optimizer,metrics=None):
        self.model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
        
    def fit(self,X,y):
        self.model.fit(X,y,batch_size=self.batch_size,epochs=self.epochs)
        
    
    def predict(self,X):
        return self.model(X)

    def get_tf_model(self,model,init_params,compile_params):
        
        init_model=model(**init_params)
        init_model.compile(**compile_params)

        return init_model
