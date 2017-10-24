#-*-coding:utf-8-*-

import csv

class CourseInfo:
    def __init__(self, filename):
        self.course_enrollment_info = {}
        with open(filename, 'r') as f:
            info = list(csv.reader(f))[1:]
            for row in info:
                enrollment_id, username, course_id = row[0:3]
                if(course_id not in self.course_enrollment_info):
                    self.course_enrollment_info[course_id] = [enrollment_id]
                else:
                    self.course_enrollment_info[course_id].append(enrollment_id)