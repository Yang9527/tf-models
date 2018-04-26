class ConfigUtil(object):
    def __init__(self, config, section="DEFAULT"):
        self.config = config
        self.sec = section

    def get(self, name):
        return self.config.get(self.sec, name)

    def get_int(self, name):
        return self.config.getint(self.sec, name)

    def get_float(self, name):
        return self.config.getfloat(self.sec, name)

    def get_boolean(self, name):
        return self.config.getboolean(self.sec, name)

    def items(self):
        return self.config.items(self.sec)

    def get_list(self, name, sep=",", dtype=str):
        lst = self.config.get(self.sec, name).split(sep)
        return [dtype(item) for item in lst]