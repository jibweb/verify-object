import argparse
from config.config import GlobalConfig
import pprint


if __name__ == "__main__":
    import yaml
    cfg = GlobalConfig()
    # import pdb; pdb.set_trace()
    cfg.optim.learning_rate = 0.01
    print(cfg.optim.learning_rate, "should be 0.01")

    cfg.from_dict({"optim": {"learning_rate": 0.02}})
    print(cfg.optim.learning_rate, "should be 0.02")
    cfg.from_dict({"optim.learning_rate": 0.03}, flat=True)
    print(cfg.optim.learning_rate, "should be 0.03")

    try:
        cfg.other_attr = 2
        print("!!! Uncaught attr creation error")
    except Exception as e:
        print("Expected error about attribute creation //", e)

    try:
        cfg.optim_cfg.mask_color = 2
        print("!!! Uncaught attr type error")
    except Exception as e:
        print("Expected error about Enum value type //", e)

    try:
        cfg.optim_cfg.learning_rate = "s"
        print("!!! Uncaught attr type error")
    except Exception as e:
        print("Expected error about value type //", e)



    print("\n", "-"*10)
    dumped_str = yaml.dump(cfg)
    print(dumped_str)

    print("\n", "-"*10)
    print(yaml.load(dumped_str, Loader=yaml.Loader))

    print("\n", "-"*10)

    dict_repr = cfg.to_dict()
    pprint.pprint(dict_repr)
    dict_repr['optim']['learning_rate'] = 0.04
    cfg.from_dict(dict_repr)
    print(cfg.optim.learning_rate, "should be 0.04")

    try:
        dict_repr['optim']['learning_rae'] = 0.04
        cfg.from_dict(dict_repr)
        print("!!! Uncaught attr creation error")
    except Exception as e:
        print("Expected error about attribute creation //", e)


    flat_dict_repr = cfg.to_dict(flat=True, sep='-')
    pprint.pprint(flat_dict_repr)
    flat_dict_repr['optim-learning_rate'] = 0.05
    cfg.from_dict(flat_dict_repr, flat=True, sep='-')
    print(cfg.optim.learning_rate, "should be 0.05")


    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    for param_name, default_val in cfg.to_dict(flat=True).items():
        parser.add_argument('--' + param_name, default=default_val, type=type(default_val))

    cfg.from_dict(vars(parser.parse_args()), flat=True)
    print(cfg.optim.learning_rate, "should be what's given in argument")
