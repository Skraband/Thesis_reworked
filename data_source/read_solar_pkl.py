import pickle
import datetime

from data_source.data_source import DataSource

class ReadSolarPKL(DataSource):

    def __init__(self, max_amt_groups=None):
        super(ReadSolarPKL, self).__init__(is_remote_source=False, plot_single_group_detailed=True)

        self._data = None
        self.company_mapping = {}
        self.max_amt_groups = max_amt_groups

    @property
    def data(self):
        # Lazy data loading
        if self._data is None:
            with open('res/solar_data/data_solar.pickle', 'rb') as f:
                data = pickle.load(f)
                self._data = data

        return self._data

    def get_identifier(self):
        # __class__.__name__
        # Or rather as attribute?
        return __class__.__name__