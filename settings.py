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
