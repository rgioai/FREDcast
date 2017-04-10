class Settings(object):
    def load(self):
        self.data = {}

        with open('_SETTINGS.txt') as settings_file:
            for line in settings_file:
                if not line.strip().startswith('#'):
                    if 'auth_code:' in line.strip():
                        self.data['auth_code'] = line.strip().split(':')[1]
                    elif 'start:' in line.strip():
                        self.data['start'] = int(line.strip().split(':')[1])
                    elif 'end:' in line.strip():
                        self.data['end'] = int(line.strip().split(':')[1])
                    elif 'unemployment:' in line.strip():
                        self.data['unemployment'] = line.strip().split(':')[1]
                    elif 'payroll:' in line.strip():
                        self.data['payroll'] = line.strip().split(':')[1]
                    elif 'gdp:' in line.strip():
                        self.data['gdp'] = line.strip().split(':')[1]
                    elif 'cpi:' in line.strip():
                        self.data['cpi'] = line.strip().split(':')[1]
                    # ISSUE Very memory inefficient, use np arrays instead of python lists
                    elif 'features:' in line.strip():
                        features = []
                        with open(str(line.strip().split(':')[1])) as f:
                            for line in f:
                                features.append(line.strip().split(','))
                        self.data['features'] = features

    def get(self, key):
        # s = Settings()
        # s.load()
        # s[key] (should work just as s.get(key)
        if self.data is None:
            self.load()
        return self.data[key]

    def set(self, key, value):
        # s = Settings()
        # s.load()
        # s[key] = value (should work just as s.set(key, value)
        if self.data is None:
            self.load()
        self.data[key] = value
