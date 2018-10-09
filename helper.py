class YearData:    
    def __init__(self, year, start_index, days):
        self.year = year
        self.start_index = start_index
        self.days = days

year_2009 = YearData(2009, 0, 365)
year_2010 = YearData(2010, 365, 365)
year_2011 = YearData(2011, 365*2, 365)
year_2012 = YearData(2012, 365*3, 366)
year_2013 = YearData(2013, 365*4 + 1, 365)
year_2014 = YearData(2014, 365*3 + 1, 365)
    
by_year = [year_2009, year_2010, year_2011, year_2012, year_2013, year_2014]

n_train_days = 4*365
n_val_days = 365
n_test_days = 365

h_files = ["water_level/Kamanje H.csv", "water_level/Karlovac H.csv", "water_level/Jamnicka kiselica H.csv", "water_level/Farkasic H.csv"]
q_files = ["flow/Kamanje Q.csv", "flow/Jamnicka kiselica Q.csv", "flow/Farkasic Q.csv"]

w_files = ["waterfall/bosiljevo.csv", "waterfall/delnice.csv", "waterfall/karlovac.csv", "waterfall/lokve.csv", "waterfall/ogulin.csv", "waterfall/parg.csv", "waterfall/pisarovina.csv", "waterfall/plaski.csv", "waterfall/slunj.csv"]
