class UserDataSet:
    
    def __init__(self):
        self.dummy=-1

    def users(self):
        ALL_USERS_FILENAME="userids.csv"
        import csv
        csvFile=open(ALL_USERS_FILENAME, newline="")
        reader=csv.reader(csvFile)
        return next(reader)