from models.gan5 import create_gen, create_disc


def selected_models(config):
    
    if config["models"] =="gan":
        return create_gen(config["training"]["batch_size"]),create_disc()