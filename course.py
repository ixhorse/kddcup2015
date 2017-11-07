#-*-coding:utf-8-*-

import csv
import datetime


class CourseInfo:
    def __init__(self, filename, date):
        self.course_enrollment_info = {}
        self.course_date = {}
        with open(filename, 'r') as f:
            info = list(csv.reader(f))[1:]
            for row in info:
                enrollment_id, username, course_id = row[0:3]
                if course_id not in self.course_enrollment_info:
                    self.course_enrollment_info[course_id] = [enrollment_id]
                else:
                    self.course_enrollment_info[course_id].append(enrollment_id)
        with open(date, 'r') as d:
            date = list(csv.reader(d))[1:]
            for row2 in date:
                course_id2, begin, to = row2[0:3]
                d = [datetime.datetime.strptime(begin, '%Y-%m-%d'), datetime.datetime.strptime(to, '%Y-%m-%d')]
                if course_id2 not in self.course_date:
                    self.course_date[course_id2] = d



#course_enrollment = CourseInfo('../data/train/enrollment_train.csv', './date.csv').course_date
#for key in course_enrollment:
#    print(course_enrollment[key][1])