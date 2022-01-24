import pandas as pd
import numpy as np
# np.random.seed(5)


class Users:
    """ User group characteristics """

    def __init__(self, city):
        assert isinstance(city, str)
        self.city_scaling = {'SiouxFalls': 0.5}  # scale user flows to ensure feasibility
        self.city = city
        self.raw_od = pd.read_csv("Locations/" + city + "/od.csv")
        self.num_users = None
        self.data = self._generate_users()

    def _generate_users(self):
        #   # to ensure vot is the same everytime we draw it

        splits = 1
        df = pd.DataFrame(np.repeat(self.raw_od.values, splits, axis=0), columns=self.raw_od.columns)
        self.num_users = splits * df.shape[0]

        # df['volume'] = round(df['volume'] / splits)
        df['volume'] = round(self.city_scaling[self.city] * df['volume']/splits)

        # TODO: draw volume from a distribution to go beyond theory!
        # Devansh thinks: Results might hold if total num of users is fixed!

        df['vot'] = self.vot_realization()

        df.rename(columns={"origin": "orig", "destination": "dest", "volume": "vol"}, inplace=True)
        data = df.to_dict('index')
        # print("Total flow volume = ", df['vol'].sum())
        return data

    def vot_list(self):
        vot = [self.data[i]['vot'] for i in range(len(self.data))]
        return vot

    def user_flow_list(self):
        user_flow = [self.data[i]['vol'] for i in range(len(self.data))]
        return user_flow

    def vot_realization(self):
        # TODO: FIXME
        vot_array = 0.9 * np.ones(self.num_users) + 0.2 * np.random.rand(self.num_users)
        # vot_array = 0.6 * np.ones(self.num_users) + 0.8 * np.random.rand(self.num_users)
        # vot_array = np.ones(self.num_users)
        return vot_array

    def new_instance(self):
        self.data = self._generate_users()
        return None

