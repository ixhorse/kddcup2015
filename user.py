#-*-coding:utf-8-*-

import csv


class UserInfo:
    def __init__(self, filename):
        self.user_enrollment_info = {}
        self.enrollment_train=[]
        with open(filename, 'r') as f:
            info = list(csv.reader(f))[1:]
            self.enrollment_train=info
            for row in info:
                enrollment_id, username= row[0:2]
                if(username not in self.user_enrollment_info):
                    self.user_enrollment_info[username] = [enrollment_id]
                else:
                    self.user_enrollment_info[username].append(enrollment_id)


