from model.base import SO, TO
from model.ecbnn import ECBNN

def get_model(model):
    model_pool = {
        'SO': SO,
        'TO': TO,
        'ECBNN': ECBNN,
    }
    return model_pool[model]